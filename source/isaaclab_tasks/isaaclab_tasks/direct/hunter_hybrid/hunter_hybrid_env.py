from __future__ import annotations

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.managers import SceneEntityCfg, EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
import numpy as np
import random
import scipy.linalg as la
import math
# from isaacsim.util.debug_draw import _debug_draw
# draw = _debug_draw.acquire_debug_draw_interface()
from isaaclab_assets.robots.hunter import HUNTER_CFG
from .CubicSpline import calc_spline_course

# Load racing track waypoints.
coordinates = np.genfromtxt(
    '/home/lee/Hybrid_DRL_Deployments/source/isaaclab_tasks/isaaclab_tasks/direct/hunter_hybrid/Austin_centerline2.csv',
    delimiter=','
)
x_coords = coordinates[::10, 0]
y_coords = coordinates[::10, 1]
coordinates = np.stack((x_coords, y_coords), axis=-1)

# --- Elevation Functions ---
def world_height_map(env, sensor_cfg: SceneEntityCfg, offset: int, plane_init_value: int):
    height_scan = -mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2] - plane_init_value
    return height_scan + world_pos_z.unsqueeze(-1)

def higher_elevation(env):
    pos = mdp.root_pos_w(env)
    z_value = pos[..., 2] - 0.19
    vel = mdp.base_lin_vel(env)[..., 0]
    condition = (z_value > 0.1) & (vel > 0.1)
    rew = torch.where(condition, z_value, torch.zeros_like(z_value))
    return torch.clip(rew, min=0, max=1)

def steep_penalty(env, thresh_pitch):
    orient = mdp.root_quat_w(env)
    euler_xyz = mdp.euler_xyz_from_quat(orient)
    euler_xyz = torch.stack(euler_xyz, dim=-1)
    pitch = euler_xyz[:, 1]
    return torch.clamp(pitch - thresh_pitch, min=0)

def randomize_vehicle_properties(env):
    """Randomize friction, mass, and initial x-y start position."""
    env.static_friction = random.uniform(0.8, 1.5)
    env.dynamic_friction = random.uniform(0.6, 1.2)
    env.vehicle_mass = random.uniform(0.9, 1.2)
    print(f"Randomized friction: static={env.static_friction:.2f}, dynamic={env.dynamic_friction:.2f}")
    print(f"Randomized mass multiplier: {env.vehicle_mass:.2f}")

def randomize_start_position(default_root_state):
    """Apply a random x-y offset to the initial root state."""
    # Randomize x and y offsets within ±2 meters.
    offset = np.random.uniform(-2.0, 2.0, size=(default_root_state.shape[0], 2))
    default_root_state[:, :2] += torch.tensor(offset, dtype=default_root_state.dtype, device=default_root_state.device)
    return default_root_state

# --- Merged Environment Configuration ---
@configclass
class HunterHybridEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        dt=1/200,
        render_interval=20,
        use_fabric=True,
        enable_scene_query_support=False,
        disable_contact_processing=False,
        gravity=(0.0, 0.0, -9.81)
    )
    physics_material: RigidBodyMaterialCfg = RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0
    )
    physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        max_position_iteration_count=4,
        max_velocity_iteration_count=0,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
        enable_stabilization=True,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=5*2**15,
        gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**21,
        gpu_heap_capacity=2**26,
        gpu_temp_buffer_capacity=2**24,
        gpu_max_num_partitions=8,
        gpu_max_soft_body_contacts=2**20,
        gpu_max_particle_contacts=2**20,
    )
    # Robot (racing agent)
    robot: ArticulationCfg = HUNTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # Scene settings: Keep the racing scene; elevation effects are added via rewards and observations.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)
    # Increase observation space by one element for elevation.
    decimation = 20
    episode_length_s = 200
    action_scale = 1
    action_space = 2
    observation_space = 8
    state_space = 0

# --- Merged Environment Class ---
class HunterHybridEnv(DirectRLEnv):
    cfg: HunterHybridEnvCfg

    def __init__(self, cfg: HunterHybridEnvCfg, render_mode: str | None = None, **kwargs):
        # Set _device before calling super().__init__ so it's available in _setup_scene.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(cfg, render_mode, **kwargs)
        # Use a local attribute _device to avoid conflict with a read-only 'device' property.
        # Get joint indices.
        self._leftwheel_dof_idx, _ = self.hunter.find_joints("re_left_jiont")
        self._rightwheel_dof_idx, _ = self.hunter.find_joints("re_right_jiont")
        self._fsr_dof_idx, _ = self.hunter.find_joints("fr_steer_left_joint")
        self._fsl_dof_idx, _ = self.hunter.find_joints("fr_steer_right_joint")
        self.joint_pos = self.hunter.data.joint_pos
        self.joint_vel = self.hunter.data.joint_vel

        # Apply domain randomization.
        randomize_vehicle_properties(self)

    def _setup_scene(self):
        # Create the vehicle articulation.
        self.hunter = Articulation(self.cfg.robot)
        # Calculate the racing spline from waypoints.
        self.cx, self.cy, self.cyaw, self.ck, _ = calc_spline_course(x_coords, y_coords, ds=0.1)
       
        # Add a ground plane.
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Clone and filter environments.
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # Add the vehicle.
        self.scene.articulations["hunter"] = self.hunter

        # Arrange environments using translations.
        self._num_per_row = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / self._num_per_row)
        num_cols = np.ceil(self.num_envs / num_rows)
        env_spacing = 5.0
        row_offset = 0.5 * env_spacing * (num_rows - 1)
        col_offset = 0.5 * env_spacing * (num_cols - 1)
        coordinates2 = np.stack((self.cx, self.cy), axis=-1)
        coordinates_tensor = torch.tensor(coordinates2, dtype=torch.float32, device=self._device)
        self.cyaw_torch = torch.tensor(self.cyaw, dtype=torch.float32, device=self._device)
        translations = []
        for i in range(self.num_envs):
            row = i // num_cols
            col = i % num_cols
            x = row_offset - row * env_spacing
            y = col * env_spacing - col_offset
            translations.append([x, y])
        translations_array = np.array(translations)
        translations_tensor = torch.tensor(translations_array, dtype=torch.float32, device=self._device)
        self.translated_coordinates = coordinates_tensor.unsqueeze(0) + translations_tensor.unsqueeze(1)
        
        # Add a dome light.
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # LQR parameters (if using an LQR controller).
        self.Q = np.eye(4)
        self.Q[0,1] = 10.0
        self.Q[1,2] = 100.0
        self.Q[2,3] = 100.0
        self.R = np.eye(1)
        self.dt = 0.005
        self.L = 0.608
        self.max_iter = 150
        self.eps = 0.0167
        self.step_counter = 0.0

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Scale and shift actions for racing.
        self.hunter_position = self.hunter.data.root_pos_w
        self.vel = self.hunter.data.root_lin_vel_b
        self.actions[:,0] = (actions[:,0] + 1.0) / 2.0 * 21.82  # Scale velocity.
        self.actions[:,1] = 0.524 * actions[:,1].clone()          # Scale steering.

    def _apply_action(self) -> None:
        # Compute steering angles for inner and outer wheels.
        self.delta_out = torch.atan(0.608 * torch.tan(self.actions[:,1]) /
                                    (0.608 + 0.5 * 0.554 * torch.tan(self.actions[:,1])))
        self.delta_in = torch.atan(0.608 * torch.tan(self.actions[:,1]) /
                                   (0.608 - 0.5 * 0.554 * torch.tan(self.actions[:,1])))
        front_right_steer = torch.where(self.actions[:,1] <= 0, self.delta_in, self.delta_out)
        front_left_steer = torch.where(self.actions[:,1] > 0, self.delta_in, self.delta_out)
        self.hunter.set_joint_position_target(front_right_steer.unsqueeze(-1), joint_ids=self._fsr_dof_idx)
        self.hunter.set_joint_position_target(front_left_steer.unsqueeze(-1), joint_ids=self._fsl_dof_idx)
        self.hunter.set_joint_velocity_target(self.actions[:,0].unsqueeze(-1), joint_ids=self._leftwheel_dof_idx)
        self.hunter.set_joint_velocity_target(self.actions[:,0].unsqueeze(-1), joint_ids=self._rightwheel_dof_idx)

    def _get_observations(self) -> dict:
        self.heading_angle_hunter = self.hunter.data.heading_w
        self.translated_coordinates = self.translated_coordinates.to(self._device)
        self.cyaw_torch = self.cyaw_torch.to(self._device)
        coordinates_hunter = self.hunter.data.root_pos_w[:, 0:2].unsqueeze(1)
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        self.min_distance_idx = torch.argmin(distances, dim=1)
        self.crosstrack_error = distances[torch.arange(distances.shape[0]), self.min_distance_idx]
        self.heading_angle_Error = self.heading_angle_hunter - self.cyaw_torch[self.min_distance_idx]
        self.heading_angle_Error = (self.heading_angle_Error + torch.pi) % (2 * torch.pi) - torch.pi
        self.crosstrack_error = torch.where(self.heading_angle_Error <= 0.0, -1.0 * self.crosstrack_error, self.crosstrack_error)
        # Elevation: get base height.
        base_height = self.hunter.data.root_pos_w[:, 2].unsqueeze(-1)
        # Get roll and yaw from quaternion.
        quat_hunter = self.hunter.data.root_quat_w
        roll, _, yaw = euler_xyz_from_quat(quat_hunter)
        roll = roll.unsqueeze(-1)
        yaw = yaw.unsqueeze(-1)
        lin_vel = self.hunter.data.root_lin_vel_b[:, 0].unsqueeze(-1)
        # Concatenate: [x, y, cross-track error, heading error, roll, yaw, base height, linear velocity]
        obs = torch.cat([
            self.hunter.data.root_pos_w[:, 0:2],
            self.crosstrack_error.unsqueeze(-1),
            self.heading_angle_Error.unsqueeze(-1),
            roll,
            yaw,
            base_height,
            lin_vel
        ], dim=-1)
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # Racing reward: path tracking.
        heading_angle_hunter = self.hunter.data.heading_w
        self.cyaw_torch = self.cyaw_torch.to(self._device)
        coordinates_hunter = self.hunter.data.root_pos_w[:, 0:2].unsqueeze(1)
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(distances, dim=1)
        heading_angle_Error = torch.abs((heading_angle_hunter - self.cyaw_torch[min_distance_idx] + torch.pi) % (2 * torch.pi) - torch.pi)
        crosstrack_error = torch.abs(distances[torch.arange(distances.shape[0]), min_distance_idx]) / 5.0
        base_lin_vel = torch.abs(self.hunter.data.root_lin_vel_b[:, 0]) / 3.0
        reward_racing = torch.exp(-1.0 * crosstrack_error) * torch.exp(-1.0 * heading_angle_Error) * (0.1 * base_lin_vel)
        # Elevation reward: bonus for higher elevation.
        base_height = self.hunter.data.root_pos_w[:, 2] - 0.19
        reward_elevation = torch.clamp(base_height, min=0.0, max=1.0) * 5000.0
        reward_total = reward_racing + reward_elevation - 1.0 * getattr(self, "hunter_reset", 0)
        reward_total = torch.clip(reward_total, min=0.0)
        return reward_total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates_hunter = self.hunter.data.root_pos_w[:, 0:2].unsqueeze(1)
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(distances, dim=1)
        crosstrack_error2 = distances[torch.arange(distances.shape[0]), min_distance_idx]
        base_lin = self.hunter.data.root_lin_vel_b[:, 0]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        hunter_reset_vel = torch.where(base_lin <= 0.01, 1.0, 0.0).bool()
        hunter_reset_crosstrack = torch.where(crosstrack_error2 >= 5.0, 1.0, 0.0).bool()
        # Terminate if the vehicle falls (base height too low).
        base_height = self.hunter.data.root_pos_w[:, 2]
        fallen = base_height < 0.1
        self.hunter_reset = hunter_reset_vel | hunter_reset_crosstrack | fallen
        return self.hunter_reset, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.hunter._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.hunter.data.default_joint_pos[env_ids].to(self._device)
        joint_vel = self.hunter.data.default_joint_vel[env_ids].to(self._device)
        default_root_state = self.hunter.data.default_root_state[env_ids].to(self._device)
        # Apply random x-y offset to starting position.
        default_root_state = randomize_start_position(default_root_state)
        default_root_state[:, :3] += self.scene.env_origins[env_ids].to(self._device)
        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel
        self.hunter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.hunter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.hunter.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
