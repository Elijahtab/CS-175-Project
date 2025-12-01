# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.envs import ViewerCfg 


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(-4.0, 1.5, 2.0),      # behind & slightly to the left
        lookat=(0.0, 0.0, 0.7),    # look toward robot torso / stairs
        resolution=(1920, 1080),
        origin_type="env",
        env_index=0,
        cam_prim_path="/OmniverseKit_Persp",
    )
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        if hasattr(self.rewards, "undesired_contacts"):
            sensor_cfg = self.rewards.undesired_contacts.params["sensor_cfg"]
            # Penalize contacts on torso + upper legs, not feet
            sensor_cfg.body_names = ("base", ".*_thigh", ".*_calf")
            sensor_cfg.preserve_order = False
        # ---------- MILESTONE 1 COMMAND RANGES (EASY FLAT) ----------
        # Forward only, small speed, no strafing, no turning
        # self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.0, 0.0),
        #     lin_vel_y=(0.0, 0.0),
        #     ang_vel_z=(0.0, 0.0),
        #     heading=(-0.0, 0.0),
        # )
        
        # ---------- MILESTONE 2 COMMAND RANGES (FASTER FLAT) ----------
        # # Allow more speed + a bit of turning
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.5, 0.5),
            heading=(-1.57, 1.57),
        )
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.heading   = (-1.57, 1.57)  # +/- ~90deg
        # override rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight  = 1.0

        # Strong posture rewards
        if hasattr(self.rewards, "flat_orientation_l2"):
            self.rewards.flat_orientation_l2.weight = -5.0

        # NEW: strong penalty for being too low / too high
        if hasattr(self.rewards, "base_height_l2"):
            # tune weight and target_height (in custom_rewards) together
            self.rewards.base_height_l2.weight = -5.0

        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -0.5

        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.1

        # Penalize non-foot contacts (as in last message)
        if hasattr(self.rewards, "undesired_contacts"):
            self.rewards.undesired_contacts.weight = -5.0
            sensor_cfg = self.rewards.undesired_contacts.params["sensor_cfg"]
            # Single regex that matches base, head, hips, thighs, calves
            sensor_cfg.body_names = "(base|Head_.*|.*_hip|.*_thigh|.*_calf)"
            sensor_cfg.preserve_order = False

        # Don’t encourage stepping yet for M1
        if hasattr(self.rewards, "feet_air_time"):
            self.rewards.feet_air_time.weight = 0.2
        
        if hasattr(self.rewards, "dof_pos_limits"):
            self.rewards.dof_pos_limits.weight = -0.1  # it’s 0.0 in your logs
        
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        
        self.scene.robot.visual_material = sim_utils.PreviewSurfaceCfg(
            func=sim_utils.spawn_preview_surface,
            diffuse_color=(0.1, 0.4, 0.9),  # bluish
            roughness=0.4,
            metallic=0.1,
        )


class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(-4.0, 1.5, 2.0),      # behind & slightly to the left
        lookat=(0.0, 0.0, 0.7),    # look toward robot torso / stairs
        resolution=(1920, 1080),
        origin_type="env",
        env_index=0,
        cam_prim_path="/OmniverseKit_Persp",
    )
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        self.scene.robot.visual_material = sim_utils.PreviewSurfaceCfg(
            func=sim_utils.spawn_preview_surface,
            diffuse_color=(0.1, 0.4, 0.9),  # bluish
            roughness=0.4,
            metallic=0.1,
        )
