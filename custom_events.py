import torch
import math
from isaaclab.envs import ManagerBasedRLEnv

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
    obj_pose = obstacles.data.object_link_pose_w.clone()[env_ids][:, :, :]  # (N, num_objs, 7)

    # base position per env, expanded to match objects
    base_xy = base_pos_w[:, :2].unsqueeze(1)  # (N, 1, 2) -> (N, num_objs, 2) via broadcast

    # set positions for active obstacles
    obj_pose[..., 0] = base_xy[..., 0] + offset_x
    obj_pose[..., 1] = base_xy[..., 1] + offset_y
    # keep z and orientation from default (obj_pose[...,2:7])

    # For inactive obstacles, you might want to move them far away or underground
    inactive_mask = ~active_mask
    if inactive_mask.any():
        # push inactive obstacles below ground
        obj_pose[..., 2] = torch.where(
            inactive_mask,
            torch.full_like(obj_pose[..., 2], -10.0),
            obj_pose[..., 2],
        )

    # Write poses back to sim
    obstacles.write_object_pose_to_sim(
        object_pose=obj_pose,
        env_ids=env_ids,
        object_ids=None,  # all objects
    )
