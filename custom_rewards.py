import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply_inverse
import isaaclab_tasks.manager_based.locomotion.velocity.config.go2_nav.custom_obs as custom_obs


# ---------------------------------------------------------
# Existing NAV rewards (keep)
# ---------------------------------------------------------
def heading_alignment(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]

    # robot yaw
    _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

    # direction to goal (world frame)
    pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    vec = goals - pos
    goal_yaw = torch.atan2(vec[:, 1], vec[:, 0])

    # relative angle (wrap)
    rel = wrap_to_pi(goal_yaw - yaw)

    # reward large when facing goal (cosine smooth)
    return torch.cos(rel)


def vel_toward_goal(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]
    vel_b = robot.data.root_lin_vel_b[:, :2]   # vx, vy in body frame

    # direction to goal in body frame
    goal_b = custom_obs.goal_direction_body(env, asset_cfg, command_name)
    goal_b = goal_b / (goal_b.norm(dim=1, keepdim=True) + 1e-6)

    # dot product = projected velocity toward goal
    reward = (vel_b * goal_b).sum(dim=1)

    # only positive forward
    return torch.clamp(reward, min=0.0)


def progress_to_goal(env, asset_cfg: SceneEntityCfg, command_name):
    robot = env.scene[asset_cfg.name]

    pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]
    dist_now = torch.norm(goals - pos, dim=-1)

    dist_prev = env.extras.get("pg_dist_prev", None)
    if dist_prev is None:
        env.extras["pg_dist_prev"] = dist_now.clone()
        return torch.zeros_like(dist_now)

    reward = dist_prev - dist_now
    env.extras["pg_dist_prev"] = dist_now.clone()

    return reward


# =========================================================
# NEW BASE LOCOMOTION REWARDS
# =========================================================

# 1. Linear velocity tracking (exp(-||v_ref - v||^2))
def lin_vel_tracking_exp(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Exponential tracking of commanded (vx, vy) in BODY frame."""
    robot = env.scene[asset_cfg.name]

    # commanded [vx_ref, vy_ref, wz_ref, heading] from command manager
    cmd = env.command_manager.get_command(command_name)
    v_ref = cmd[:, :2]  # (vx_ref, vy_ref)

    # actual base linear velocity in body frame: (vx, vy)
    v = robot.data.root_lin_vel_b[:, :2]

    diff = v_ref - v
    sq = torch.sum(diff * diff, dim=1)
    
    return torch.exp(-sq)


# 2. Angular velocity tracking (exp(-(wz_ref - wz)^2))
def ang_vel_tracking_exp(env, asset_cfg: SceneEntityCfg, command_name: str):
    """Exponential tracking of commanded yaw rate (wz)."""
    robot = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command(command_name)
    wz_ref = cmd[:, 2]  # assuming [vx, vy, wz, heading]

    wz = robot.data.root_ang_vel_b[:, 2]  # yaw rate in body frame

    diff = wz_ref - wz
    return torch.exp(-(diff * diff))

# 3. Height error (z - z_ref)^2  (use constant z_ref from params)
def base_height_error(env, asset_cfg: SceneEntityCfg, target_height: float = 0.38, asset_radius: float = 0.1):
    """
    Penalize deviation from target height RELATIVE TO TERRAIN.
    """
    # 1. Get Robot Data
    robot = env.scene[asset_cfg.name]

    #ORIGINAL:
    # z = robot.data.root_pos_w[:, 2]
    # diff = z - target_height
    # return diff * diff  # use NEGATIVE weight to make it a penalty

    #NEW:
    root_pos_w = robot.data.root_pos_w
    # 2. Get Terrain Height at Robot's (X, Y)
    # We query the terrain generator for the height under the robot's center
    # Note: Requires env.scene.terrain to be active
    terrain_heights = env.scene.terrain.terrain_height_at_pos(root_pos_w[:, :2])
    
    # 3. Calculate Relative Height (Robot Z - Ground Z)
    # We subtract terrain height to get 'local' height
    current_height = root_pos_w[:, 2] - terrain_heights
    
    # 4. Calculate Error
    # FUTURE PROOFING: Later, replace 'target_height' with: 
    # target = env.command_manager.get_command("gait_params")[:, 0]
    diff = current_height - target_height
    
    return torch.square(diff)


def track_commanded_height_exp(env, command_name: str, asset_cfg: SceneEntityCfg):
    """
    Reward for matching the commanded base height relative to the terrain.
    Uses an exponential kernel: exp(-error^2 / std^2).
    """
    # 1. Get the Command (The "Target")
    # We fetch the command named 'gait_params'. 
    # Assumes format: [Body Height, Step Freq, Clearance]
    # We only want index 0 (Body Height).
    gait_cmd = env.command_manager.get_command(command_name)
    target_height_offset = gait_cmd[:, 0] # This is usually an offset, e.g., -0.1m
    
    # Define the nominal standing height (The "Zero" point)
    # You can pass this as a param, or hardcode it if your robot is fixed.
    nominal_height = 0.38 
    target_height = nominal_height + target_height_offset

    # 2. Get Current Height (Relative to Terrain)
    robot = env.scene[asset_cfg.name]
    root_pos_w = robot.data.root_pos_w
    
    # Handle flat ground vs rough terrain cases safely
    if env.scene.terrain.cfg.terrain_type == "plane":
        terrain_height = torch.zeros_like(root_pos_w[:, 2])
    else:
        terrain_height = env.scene.terrain.terrain_height_at_pos(root_pos_w[:, :2])
        
    current_height = root_pos_w[:, 2] - terrain_height

    # 3. Calculate Error
    error = torch.square(current_height - target_height)
    
    # 4. Convert to Reward (Exponential is better for tracking than L2 penalty)
    # This gives +1.0 for perfect tracking, decaying to 0.0 as error increases.
    std = 0.05 # Tolerance window (5cm)
    return torch.exp(-error / (std**2))


def track_feet_clearance_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sigma: float = 0.01,
    swing_vel_thresh: float = 0.1,
):
    """
    Reward for matching commanded foot clearance (command[2]) relative to local ground.

    - Uses a RayCaster (e.g. 'height_scanner') to estimate ground height.
    - Applies exp(-error / sigma) only on swinging feet.
    """
    # 1. Command: [height, freq, clearance]
    gait_cmd = env.command_manager.get_command(command_name)  # (num_envs, 3)
    target_clearance = gait_cmd[:, 2]                         # (num_envs,)

    # 2. Foot positions
    robot = env.scene[asset_cfg.name]
    # positions of the selected bodies (feet) in world frame: (num_envs, num_feet, 3)
    foot_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_z = foot_pos_w[..., 2]  # (num_envs, num_feet)

    # 3. Ground height from height scanner (RayCaster)
    sensor = env.scene[sensor_cfg.name]
    # ray_hits_w: (num_envs, num_rays, 3)
    ground_z = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)  # (num_envs,)

    # clearance = foot_z - ground_z
    clearance = foot_z - ground_z.unsqueeze(1)  # broadcast: (num_envs, num_feet)

    # 4. Error (only when below target)
    target_expand = target_clearance.unsqueeze(1)  # (num_envs, 1)
    below_target = torch.clamp(target_expand - clearance, min=0.0)
    error = below_target**2  # (num_envs, num_feet)

    # 5. Swing mask (feet moving in world)
    foot_vel = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)  # (num_envs, num_feet)
    is_swinging = (foot_vel > swing_vel_thresh).float()

    # 6. Exponential kernel + sum across feet
    rew_per_foot = torch.exp(-error / sigma) * is_swinging
    reward = torch.sum(rew_per_foot, dim=1)  # (num_envs,)

    return reward


# 4. Pose similarity ||q - q_default||^2
def pose_similarity(env, asset_cfg: SceneEntityCfg):
    """Squared distance to the robot's default joint configuration."""
    robot = env.scene[asset_cfg.name]

    q = robot.data.joint_pos

    # Many Isaac/IsaacLab articulations expose default joint pose; if not,
    # fall back to zero as a safe no-op.
    if hasattr(robot.data, "default_joint_pos") and robot.data.default_joint_pos is not None:
        q_def = robot.data.default_joint_pos
    else:
        q_def = torch.zeros_like(q)

    diff = q - q_def
    return torch.sum(diff * diff, dim=1)


# 6. Vertical velocity penalty v_z^2
def vertical_vel_penalty(env, asset_cfg: SceneEntityCfg):
    """Penalty on vertical (z-axis) linear velocity of the base."""
    robot = env.scene[asset_cfg.name]
    vz = robot.data.root_lin_vel_b[:, 2]
    return vz * vz  # use negative weight


# 7. Roll/pitch stabilization penalty (phi^2 + theta^2)
def roll_pitch_penalty(env, asset_cfg: SceneEntityCfg):
    """
    Penalty for non-flat base orientation.
    Optimization: Uses projected gravity instead of Euler angles (faster/safer).
    """
    robot = env.scene[asset_cfg.name]

    #ORIGINAL:
    # roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    # return roll * roll + pitch * pitch  # use negative weight


    #NEW:
    # The gravity vector in World frame is [0, 0, -1]
    # We rotate it into the Robot's Body frame
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    projected_gravity_b = quat_apply_inverse(robot.data.root_quat_w, gravity_vec_w)
    
    # If perfectly flat, projected_gravity_b should be [0, 0, -1]
    # The x and y components represent roll/pitch tilt
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

def forward_velocity_bonus(env, asset_cfg):
    robot = env.scene[asset_cfg.name]          # FIX: IsaacLab uses dict-style access
    base_lin_vel_b = robot.data.root_lin_vel_b # (num_envs, 3)

    forward_vel = base_lin_vel_b[:, 0]         # body-frame X velocity
    return forward_vel

def negative_forward_velocity_penalty(env, env_ids=None, asset_cfg=None):
    """
    Penalize backward motion (negative x-velocity).
    """
    robot = env.scene[asset_cfg.name]

    # Always fix env_ids to slice(None) when empty
    if env_ids is None:
        env_ids = slice(None)

    # Extract forward velocity — world frame X
    base_lin_vel = robot.data.root_vel_w[env_ids, 0]   # shape [num_envs]

    # Penalize backward motion only
    penalty = torch.clamp(-base_lin_vel, min=0.0)
    return penalty
