import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


<<<<<<< Updated upstream
def lin_vel_y_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize Y-axis (sideways) velocity."""
    asset = env.scene[asset_cfg.name]
    # asset.data.root_lin_vel_b is [num_envs, 3] (x, y, z)
    # We want index 1 (y)
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def progress_to_goal(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Reward for moving closer to the goal.
    Formula: (Old_Distance - New_Distance) + (Velocity * Goal_Direction)
    We typically use a "Potential-Based" reward: current_dist - prev_dist.
    """
    # 1. Get the robot and goals
=======
# ---------------------------------------------------------
# Existing NAV rewards (keep)
# ---------------------------------------------------------
def heading_alignment(env, asset_cfg: SceneEntityCfg, command_name):
>>>>>>> Stashed changes
    robot = env.scene[asset_cfg.name]

    # Get only the (X, Y) position [num_envs, 2]
    current_pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]

    # 2. Calculate current distance to goal
    target_vec = goals - current_pos
    current_dist = torch.norm(target_vec, dim=1)

<<<<<<< Updated upstream
    # 3. Calculate "Previous" distance
    # We approximate previous pos using velocity: prev_pos = curr_pos - (vel * dt)
    dt = env.step_dt
    # root_lin_vel_w is [num_envs, 3], we need [:, :2]
    current_vel = robot.data.root_lin_vel_w[:, :2]
    prev_pos = current_pos - (current_vel * dt)

    prev_target_vec = goals - prev_pos
    prev_dist = torch.norm(prev_target_vec, dim=1)

    # 4. Reward = Improvement in distance
    # If we got closer, prev_dist > current_dist, so result is positive.
    reward = prev_dist - current_dist
=======
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
>>>>>>> Stashed changes

    return reward


<<<<<<< Updated upstream
def arrival_reward(env: ManagerBasedRLEnv, command_name: str, threshold: float,
                   asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Sparse reward: +1.0 if within [threshold] meters of the goal.
    """
    robot = env.scene[asset_cfg.name]

    current_pos = robot.data.root_pos_w[:, :2]
    goals = env.command_manager.get_command(command_name)[:, :2]

    distance = torch.norm(goals - current_pos, dim=1)

    # Return 1.0 where distance < threshold, else 0.0
    # converting bool to float gives us 1.0 or 0.0
    return (distance < threshold).float()
=======
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
def base_height_error(env, asset_cfg: SceneEntityCfg, target_height: float):
    """Squared error of base height around a fixed target_height."""
    robot = env.scene[asset_cfg.name]
    z = robot.data.root_pos_w[:, 2]
    diff = z - target_height
    return diff * diff  # use NEGATIVE weight to make it a penalty


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
    """Penalty on roll and pitch of the base."""
    robot = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return roll * roll + pitch * pitch  # use negative weight

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
>>>>>>> Stashed changes
