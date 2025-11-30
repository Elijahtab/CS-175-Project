import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from . import custom_obs


# ---------------------------------------------------------
# 1. Heading alignment reward
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


# ---------------------------------------------------------
# 2. Velocity toward goal (body frame)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 3. Progress toward goal (world frame)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# 4. Base height penalty (L2)
# ---------------------------------------------------------
def base_height_l2(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize deviation of robot base height from a target height (L2).

    For flat terrain, we just compare to a fixed target height.
    For rough terrain, an optional sensor can be used to shift the target.
    """
    asset = env.scene[asset_cfg.name]

    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + sensor.data.pos_w[:, 2]
    else:
        adjusted_target_height = target_height

    z = asset.data.root_link_pos_w[:, 2]
    return (z - adjusted_target_height) ** 2