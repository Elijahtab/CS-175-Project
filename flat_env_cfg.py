# from isaaclab.utils import configclass
# from .rough_env_cfg import UnitreeGo2RoughEnvCfg, NavCommandsCfg
# from .commands import GoalCommandCfg

# @configclass
# class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
        
#         curriculum = None
#         if hasattr(self.curriculum, "terms"):
#             for k in list(self.curriculum.terms.keys()):
#                 if "terrain" in k:
#                     print(f"[DEBUG] Removing curriculum term: {k}")
#                     del self.curriculum.terms[k]

#         self.scene.terrain.terrain_type = "plane"


#         self.commands.goal_pos = GoalCommandCfg(debug_vis=True)
        



# @configclass
# class UnitreeGo2FlatEnvCfg_PLAY(UnitreeGo2FlatEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # Reduce env count for play mode
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5

#         # Disable corruption / pushing events
#         self.observations.policy.enable_corruption = False
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None
