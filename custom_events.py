import torch
import math
from isaaclab.envs import ManagerBasedRLEnv

# You can reuse your existing constants, or hardcode.
# If you want to reuse OBSTACLE_HEIGHT, you can either:
# - import it from rough_env_cfg, OR
# - just rely on the default z already stored in object_link_pose_w.
# I'll do the second (simpler) option.

def randomize_obstacles_static_startup(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    inner_radius: float = 1.0,   # no cubes inside this radius
    outer_radius: float = 3.0,   # max radius where cubes can appear
) -> None:
    """
    One-time placement of static obstacles per environment.

    - Called with mode=\"startup\" so it runs once when envs are created.
    - All obstacles stay kinematic & never move again.
    - We sample XY positions in an annulus [inner_radius, outer_radius].
    """
    scene = env.scene
    obstacles = scene["obstacles"]  # RigidObjectCollection

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    num_envs = env_ids.shape[0]
    num_objs = obstacles.num_objects

    # --- Sample radii in [inner_radius, outer_radius] (uniform in area) ---
    # r^2 is uniform so we don't cluster near inner_radius
    rand_u = torch.rand((num_envs, num_objs), device=env.device)
    r_inner2 = inner_radius ** 2
    r_outer2 = outer_radius ** 2
    rand_r = torch.sqrt(r_inner2 + (r_outer2 - r_inner2) * rand_u)

    # Uniform theta in [0, 2π)
    rand_theta = 2.0 * math.pi * torch.rand((num_envs, num_objs), device=env.device)

    offset_x = rand_r * torch.cos(rand_theta)
    offset_y = rand_r * torch.sin(rand_theta)

    # --- Fetch current poses and overwrite X/Y only ---
    # shape: [num_envs, num_objs, 7] (x, y, z, qw, qx, qy, qz)
    obj_pose = obstacles.data.object_link_pose_w.clone()[env_ids]

    # set positions relative to env origin (0,0); z stays as default (half-height)
    obj_pose[..., 0] = offset_x
    obj_pose[..., 1] = offset_y
    # obj_pose[..., 2] left unchanged → cubes remain sitting on ground

    obstacles.write_object_pose_to_sim(
        object_pose=obj_pose,
        env_ids=env_ids,
        object_ids=None,  # all obstacles
    )


def randomize_obstacles(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    spawn_radius: float = 3.0,
    min_gap_from_robot: float = 0.7,
    max_active_obstacles: int = 4,
    obstacle_density: float = 0.3,
) -> None:
    """Randomize obstacle positions around the robot on reset.

    env_ids: the env indices this event is being applied to (EventManager passes this in).
    Other args come from EventTerm.params.
    """
    scene = env.scene
    robot = scene["robot"]
    obstacles = scene["obstacles"]  # RigidObjectCollection

    # All envs if none given
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Get robot base positions [num_envs, 3]
    base_pos_w = robot.data.root_pos_w[env_ids]  # (N, 3)

    num_envs = env_ids.shape[0]
    num_objs = obstacles.num_objects

    # Decide how many obstacles per env (sparse by default)
    # around ~obstacle_density * num_objs, clipped by max_active_obstacles
    max_per_env = min(max_active_obstacles, num_objs)
    # sample number of active obstacles (0..max_per_env)
    # simple: Bernoulli per object, then clamp
    bern = torch.rand((num_envs, num_objs), device=env.device)
    active_mask = (bern < obstacle_density)
    # force at most max_per_env per env
    counts = active_mask.sum(dim=1, keepdim=True)
    too_many = counts > max_per_env
    # if too many, randomly trim
    if too_many.any():
        # generate random scores and keep the smallest `max_per_env`
        scores = torch.rand_like(bern)
        scores[~active_mask] = 1e9
        kth = torch.topk(scores, k=max_per_env, dim=1, largest=False).values[:, -1:]
        # new mask: only those with score <= kth and originally active
        active_mask = active_mask & (scores <= kth)

    # Sample random polar offsets in XY for each env/obstacle
    # shape: (num_envs, num_objs)
    rand_r = spawn_radius * torch.sqrt(torch.rand((num_envs, num_objs), device=env.device))
    rand_theta = 2.0 * math.pi * torch.rand((num_envs, num_objs), device=env.device)

    offset_x = rand_r * torch.cos(rand_theta)
    offset_y = rand_r * torch.sin(rand_theta)

    # Enforce min distance from robot in XY
    too_close = (rand_r < min_gap_from_robot)
    # if too close, push them out to min_gap_from_robot
    rand_r = torch.where(too_close, torch.full_like(rand_r, min_gap_from_robot), rand_r)
    offset_x = rand_r * torch.cos(rand_theta)
    offset_y = rand_r * torch.sin(rand_theta)

    # Build object poses: [N, num_objs, 7] (pos xyz, quat wxyz)
    # Start from default poses, just tweak XY
    # base position per env, expanded to match objects
    base_xy = base_pos_w[:, :2].unsqueeze(1)  # (N, 1, 2) -> (N, num_objs, 2) via broadcast

    # Build object poses: [N, num_objs, 7] (pos xyz, quat wxyz)
    obj_pose = obstacles.data.object_link_pose_w.clone()[env_ids]  # (N, num_objs, 7)

    # (x, y) for all obstacles
    obj_pose[..., 0] = base_xy[..., 0] + offset_x
    obj_pose[..., 1] = base_xy[..., 1] + offset_y

    # Now set z based on active / inactive
    z_active   = torch.full_like(obj_pose[..., 2], OBSTACLE_HEIGHT * 0.5)
    z_inactive = torch.full_like(obj_pose[..., 2], -10.0)

    obj_pose[..., 2] = torch.where(active_mask, z_active, z_inactive)

    # Write poses back to sim
    obstacles.write_object_pose_to_sim(
        object_pose=obj_pose,
        env_ids=env_ids,
        object_ids=None,  # all objects
    )
