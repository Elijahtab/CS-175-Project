import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
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
