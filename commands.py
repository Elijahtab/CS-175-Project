from dataclasses import dataclass
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_SPHERE_MARKER_CFG  # or your custom marker


# 1. The Implementation Class (Logic)
class GoalCommand(CommandTerm):
    """
    Generates a random (X,Y) goal within a radius.
    """
    cfg: "GoalCommandCfg"

    def __init__(self, cfg: CommandTermCfg, env):
        super().__init__(cfg, env)
        # Create a visualization marker
        self.marker = VisualizationMarkers(cfg.visualizer_cfg)

    def _resample_command(self, env_ids):
        """
        Called when the timer runs out or robot reaches goal.
        """
        # Sample Random Radius (1m to 5m)
        r = torch.empty(len(env_ids), device=self.device).uniform_(1.0, 5.0)
        theta = torch.empty(len(env_ids), device=self.device).uniform_(0, 2 * 3.14159)

        self.command[env_ids, 0] = r * torch.cos(theta)  # Goal X
        self.command[env_ids, 1] = r * torch.sin(theta)  # Goal Y
        self.command[env_ids, 2] = 0.0  # Heading (optional)

    def _update_command(self, dt: float):
        """
        Called every step to update visuals.
        """
        if self.cfg.debug_vis:
            self.marker.visualize(self.command[:, :3])

    def _update_metrics(self):
        # not using custom metrics
        return {}

    # ------------------------------------------------------------------
    # INTERNAL IMPLEMENTATION
    # ------------------------------------------------------------------

    def __init__(self, cfg: "GoalCommandCfg", env):
        super().__init__(cfg, env)

        self._device = self._env.device
        self._num_envs = self._env.num_envs

        # (x, y, heading) in *local* env frame
        self._command_tensor = torch.zeros(self._num_envs, 3, device=self._device)
        self._time_left = torch.zeros(self._num_envs, device=self._device)

        # for “was I externally overwritten?” debugging
        self._prev_command = self._command_tensor.clone()

        # Optional visual marker
        self._marker = VisualizationMarkers(cfg.visualizer_cfg) if cfg.debug_vis else None

        print("=== USING CUSTOM GOAL COMMAND ===")

        # Initial sample for all envs
        env_ids = torch.arange(self._num_envs, device=self._device)
        self._initial_resample(env_ids)

    # ------------------------------------------------------------------

    def _initial_resample(self, env_ids: torch.Tensor):
        """Used only at construction time."""
        n = env_ids.numel()
        if n == 0:
            return

        r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
        theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
        heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

        self._command_tensor[env_ids, 0] = r * torch.cos(theta)
        self._command_tensor[env_ids, 1] = r * torch.sin(theta)
        self._command_tensor[env_ids, 2] = heading

        self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
            *self.cfg.resampling_time_range
        )

        self._visualize()

    def _timer_resample(self, env_ids: torch.Tensor):
        """Timer-based resampling. Only called when time_left <= 0."""
        n = env_ids.numel()
        if n == 0:
            return

        r = torch.empty(n, device=self._device).uniform_(*self.cfg.radius_range)
        theta = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)
        heading = torch.empty(n, device=self._device).uniform_(-math.pi, math.pi)

        self._command_tensor[env_ids, 0] = r * torch.cos(theta)
        self._command_tensor[env_ids, 1] = r * torch.sin(theta)
        self._command_tensor[env_ids, 2] = heading

        self._time_left[env_ids] = torch.empty(n, device=self._device).uniform_(
            *self.cfg.resampling_time_range
        )

    # ------------------------------------------------------------------

    def _visualize(self):
        if not self._marker:
            return

        # Local command (N, 3)
        local = self._command_tensor

        # Env origins from IsaacLab (N, 3)
        env_origins = self._env.scene.env_origins.to(self._device)

        # Convert to world frame
        world = torch.zeros(self._num_envs, 3, device=self._device)
        world[:, :2] = local[:, :2] + env_origins[:, :2]
        world[:, 2] = 0.3   # fixed Z height

        self._marker.visualize(world)

    # ------------------------------------------------------------------

    def _compute(self, dt: float):
        # Debug: timer should start big and count down slowly
        # print("Before countdown, time_left[:10]:", self._time_left[:10])

        # countdown
        self._time_left -= dt

        # resample only when timer hits zero
        env_ids = torch.nonzero(self._time_left <= 0.0, as_tuple=False).squeeze(-1)
        if env_ids.numel() > 0:
            # print("Resampling for envs:", env_ids)
            self._timer_resample(env_ids)

        # debug: detect external overwrites (shouldn’t happen)
        #if torch.any(torch.ne(self._prev_command, self._command_tensor)):
        #    comment this out once stable if it’s spammy
        #    print("Command changed (by timer or something else). Command[0]:", self._command_tensor[0])
        self._prev_command = self._command_tensor.clone()

        # update marker positions
        self._visualize()

        # === Convert goal direction → velocity command ===

        goal_xy = self._command_tensor[:, :2]   # local frame
        goal_dist = torch.norm(goal_xy, dim=1) + 1e-6
        goal_dir = goal_xy / goal_dist.unsqueeze(1)

        desired_vx = goal_dir[:, 0] * 0.6        # 0.6 m/s forward
        desired_vy = goal_dir[:, 1] * 0.2        # small sideways allowed
        desired_yaw = self._command_tensor[:, 2] # heading target from goal

        # write into base_velocity command (through command manager)
        #self._env.command_manager.commands["base_velocity"].set_velocities(
        #    desired_vx, desired_vy, desired_yaw
        #)


# 2. The Configuration Class (Settings)
@dataclass
class GoalCommandCfg(CommandTermCfg):
    # [CRITICAL] This links the Config to the Logic class above!
    class_type: type = GoalCommand

    # Default settings
    resampling_time_range: tuple[float, float] = (5.0, 10.0)
    visualizer_cfg: object = BLUE_SPHERE_MARKER_CFG
    debug_vis: bool = True
    num_commands: int = 3
