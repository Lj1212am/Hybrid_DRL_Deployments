from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg, RigidObject
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Imu, ImuCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg
from isaaclab.managers import SceneEntityCfg
import numpy as np
import random
import scipy.linalg as la
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
# from isaacsim.util.debug_draw import _debug_draw
# draw = _debug_draw.acquire_debug_draw_interface()
import math
from isaaclab_assets.robots.hunter import HUNTER_CFG
from .LQRController import State
from .angle import angle_mod
from .CubicSpline import calc_spline_course
coordinates = np.genfromtxt('/home/lee/Hybrid_DRL_Deployments/source/isaaclab_tasks/isaaclab_tasks/direct/hunter_hybrid/Austin_centerline2.csv', delimiter = ',')
x_coords = coordinates[::10, 0]
y_coords = coordinates[::10, 1]
coordinates = np.stack((x_coords, y_coords), axis=-1) 

@configclass
class HunterHybridEnvCfg(DirectRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(
    dt=1 / 200,
    render_interval=20,
    use_fabric=True,
    enable_scene_query_support=False,
    disable_contact_processing=False,
    gravity=(0.0, 0.0, -9.81),

    physics_material=RigidBodyMaterialCfg(
        static_friction=0.8,
        dynamic_friction=0.6,
        restitution=0.0
    ),
    
    physx=PhysxCfg(
        solver_type=1,
        max_position_iteration_count=4,
        max_velocity_iteration_count=0,
        bounce_threshold_velocity=0.2,
        friction_offset_threshold=0.04,
        friction_correlation_distance=0.025,
        enable_stabilization=True,
        gpu_max_rigid_contact_count=2**23,
        gpu_max_rigid_patch_count=5 * 2**15,
        gpu_found_lost_pairs_capacity=2**21,
        gpu_found_lost_aggregate_pairs_capacity=2**25,
        gpu_total_aggregate_pairs_capacity=2**21,
        gpu_heap_capacity=2**26,
        gpu_temp_buffer_capacity=2**24,
        gpu_max_num_partitions=8,
        gpu_max_soft_body_contacts=2**20,
        gpu_max_particle_contacts=2**20,
    )
)

    # action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    # )

    # # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    # observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
    #   noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
    #   bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    # )

    # robot 
    robot: ArticulationCfg = HUNTER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    #scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)
    
    #env
    
    decimation = 20
    episode_length_s = 200
    action_scale = 1  # [N]
    action_space = 2
    observation_space = 7
    state_space = 0

class HunterHybridEnv(DirectRLEnv):
    cfg: HunterHybridEnvCfg

    def __init__(self, cfg: HunterHybridEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._leftwheel_dof_idx, _ = self.hunter.find_joints("re_left_jiont")   #DOF index of front steer right joint                           
        
        self._rightwheel_dof_idx, _ = self.hunter.find_joints("re_right_jiont")

        self._fsr_dof_idx, _ = self.hunter.find_joints("fr_steer_left_joint")   #DOF index of front steer right joint                           
        self._fsl_dof_idx, _ = self.hunter.find_joints("fr_steer_right_joint")  #DOF index of front steer left joint
        self.joint_pos = self.hunter.data.joint_pos
        self.joint_vel = self.hunter.data.joint_vel


    def _setup_scene(self):
        self.hunter = Articulation(self.cfg.robot)
     
        self.cx, self.cy, self.cyaw, self.ck, _ = calc_spline_course(
        x_coords, y_coords, ds=0.1)
       
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["hunter"] = self.hunter
        

        self._num_per_row = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / self._num_per_row)
        num_cols = np.ceil(self.num_envs / num_rows)
        env_spacing = 5.0

        row_offset = 0.5 * env_spacing * (num_rows - 1)
        col_offset = 0.5 * env_spacing * (num_cols - 1)
        coordinates2 = np.stack((self.cx, self.cy), axis=-1)
 
        coordinates_tensor = torch.tensor(coordinates2, dtype=torch.float32)
        self.cyaw_torch = torch.tensor(self.cyaw, dtype=torch.float32)

        translations = []
        

        for i in range(self.num_envs):
            # compute transform
            row = i // num_cols
            col = i % num_cols
            x = row_offset - row * env_spacing
            y = col * env_spacing - col_offset
            translations.append([x,y])
        translations_array = np.array(translations)
        translations_tensor = torch.tensor(translations_array, dtype=torch.float32)
        # Apply translations
        self.translated_coordinates = coordinates_tensor.unsqueeze(0) + translations_tensor.unsqueeze(1)
        translated_coordinates_numpy = self.translated_coordinates.numpy()
        num_envs = translated_coordinates_numpy.shape[0]
        num_points = translated_coordinates_numpy.shape[1]
        colors = [(random.uniform(1.0, 1.0), random.uniform(1.0, 1.0), random.uniform(1.0, 1.0), 1) for _ in range(num_points)]
        sizes = [5 for _ in range(num_points)]
        
        #Loop through each environment and prepare the points for drawing
        # for env_index in range(num_envs):
        #     point_list = []
            
        #     for point_index in range(num_points):
        #         # Extract x, y, and assume z as 0 for 2D points; modify if you have a z component
        #         x, y = translated_coordinates_numpy[env_index, point_index]
        #         z = 0.1  # Set z to 0, or you can include it if you have a 3D point

        #         point_list.append((x, y, z))

        # #    # Draw the path for the current environment
        #     draw.draw_points(point_list, colors, sizes)

        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.Q = np.eye(4)         # LQR controller Q and R matrix
        self.Q[0,1] = 10.0
        self.Q[1,2] = 100.0
        self.Q[2,3] = 100.0
        self.R = np.eye(1)
        self.dt = 0.005
        self.L = 0.608
        self.max_iter = 150     # LQR controller DARE maximum iterations
        self.eps = 0.0167       # Epsilon value
        self.step_counter = 0.0
        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.hunter_position = self.hunter.data.root_pos_w
        
        self.vel = self.hunter.data.root_lin_vel_b
        
        self.actions[:,0] = actions[:,0].clone()
        self.actions[:,0] -= -1.0
        self.actions[:,0] /= 2.0
        self.actions[:,0] = 21.82*self.actions[:,0]
        self.actions[:,1] = 0.524*actions[:,1].clone() 
        
        # for i in range(self.num_envs):
        #     state = State(x=self.hunter_position[i, 0].cpu().numpy(), y=self.hunter_position[i, 1].cpu().numpy(), yaw=self.heading_angle_hunter[i].cpu().numpy(), 
        #                   v=self.vel[i,0].cpu().numpy())
                

        #     # Solve Discrete Time Riccati Equation      
                
        #     x = self.Q
        #     x_next = self.Q
            
        #     v = state.v
        #     # v = state.v
            
        #     # dx = np.subtract(state.x, self.cx)
        #     # dy = np.subtract(state.y, self.cy)

        #     # d = np.square(dx) + np.square(dy)
        #     # mind = np.min(d)
        #     # self.ind = np.argmin(d)

        #     A = np.zeros((4, 4))
        #     A[0, 0] = 1.0
        #     A[0, 1] = self.dt
        #     A[1, 2] = v
        #     A[2, 2] = 1.0
        #     A[2, 3] = self.dt
        #     # print(A)

        #     B = np.zeros((4, 1))
        #     B[3, 0] = v / self.L

        #     for j in range(self.max_iter):
        #         x_next = A.T @ x @ A - A.T @ x @ B @ \
        #                 la.inv(self.R + B.T @ x @ B) @ B.T @ x @ A + self.Q
        #         if (abs(x_next - x)).max() < self.eps:
        #             break
        #         x = x_next

        #     X = x_next
        #     K = la.inv(B.T @ X @ B + self.R) @ (B.T @ X @ A)

        #     # Formalize LQR controller
        #     self.ind = self.min_distance_idx.cpu().numpy()
        #     k = self.ck[self.ind[i]]
        #     v = state.v
        #     th_e = state.yaw - self.cyaw[self.ind[i]]  ### Dont use standard heading angle error, use self.cyaw

        #     x = np.zeros((4, 1))
        #     e = k
        #     pe = 0.0
        #     pth_e = 0.0
        #     x[0, 0] = e
        #     x[1, 0] = (e - pe) / self.dt
        #     x[2, 0] = th_e
        #     x[3, 0] = (th_e - pth_e) / self.dt

        #     ff = math.atan2(self.L * k, 1)
        #     fb = (-K @ x)[0, 0]

        #     steer = ff + fb
        
        #     self.actions[i,1] = torch.clamp((steer + 0.0*actions[i,1]), min=-0.524, max=0.524)
               
    def _apply_action(self) -> None:
        self.delta_out = torch.atan(0.608*torch.tan(self.actions[:,1])/
                                    (0.608 + 0.5*0.554*torch.tan(self.actions[:,1])))
        
        self.delta_in = torch.atan(0.608*torch.tan(self.actions[:,1])/
                                    (0.608 - 0.5*0.554*torch.tan(self.actions[:,1])))
        # self.delta_in = 0.0
        # self.delta_out = 0.0
        
        front_right_steer = torch.where(self.actions[:,1]<=0, self.delta_in, self.delta_out)
        front_left_steer = torch.where(self.actions[:,1] > 0, self.delta_in, self.delta_out)
        self.hunter.set_joint_position_target(front_right_steer.unsqueeze(-1), joint_ids=self._fsr_dof_idx)
        self.hunter.set_joint_position_target(front_left_steer.unsqueeze(-1), joint_ids=self._fsl_dof_idx)
        self.hunter.set_joint_velocity_target(self.actions[:,0].unsqueeze(-1), joint_ids=self._leftwheel_dof_idx)
        self.hunter.set_joint_velocity_target(self.actions[:,0].unsqueeze(-1), joint_ids=self._rightwheel_dof_idx)

    def _get_observations(self) -> dict:
        
        self.heading_angle_hunter = self.hunter.data.heading_w # Yaw -pi to pi
        self.translated_coordinates = self.translated_coordinates.to(self.device)
        self.cyaw_torch = self.cyaw_torch.to(self.device)
        self.previous_actions = self.actions.clone()
        coordinates_hunter = self.hunter.data.root_pos_w[:,0:2].unsqueeze(1)
            
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        self.min_distance_idx = torch.argmin(distances, dim=1)
        
        # min_distance_idx_1 = torch.clamp(self.min_distance_idx+1, min=0, max=self.translated_coordinates.size(1) - 1)
        # desired_heading = self.translated_coordinates[torch.arange(distances.shape[0]), min_distance_idx_1] - \
        #                   self.translated_coordinates[torch.arange(distances.shape[0]), self.min_distance_idx]
        # desired_heading_angle = torch.atan2(desired_heading[:,1], desired_heading[:,0])
        # self.heading_angle_Error = desired_heading_angle - self.heading_angle_hunter

        self.crosstrack_error = distances[torch.arange(distances.shape[0]), self.min_distance_idx]
        self.heading_angle_Error = -self.cyaw_torch[self.min_distance_idx] + self.heading_angle_hunter
        self.heading_angle_Error = (self.heading_angle_Error + torch.pi)%(2*torch.pi) - torch.pi
        self.crosstrack_error = torch.where(self.heading_angle_Error <= 0.0, -1.0*self.crosstrack_error, self.crosstrack_error)
     
        quat_hunter = self.hunter.data.root_quat_w
        roll, _, yaw = euler_xyz_from_quat(quat_hunter) # Convert to -pi to pi

        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self.hunter.data.root_pos_w[:,0:2],
                    self.crosstrack_error.unsqueeze(-1),
                    self.heading_angle_Error.unsqueeze(-1),
                    roll.unsqueeze(-1),
                    yaw.unsqueeze(-1),
                    self.hunter.data.root_lin_vel_b[:,0].unsqueeze(-1),

                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        
        return observations

    def _get_rewards(self) -> torch.Tensor:
        #self.translated_coordinates = self.translated_coordinates.to(self.device)
        heading_angle_hunter = self.hunter.data.heading_w # Yaw -pi to pi
        self.cyaw_torch = self.cyaw_torch.to(self.device)
        
        coordinates_hunter = self.hunter.data.root_pos_w[:,0:2].unsqueeze(1)    
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(distances, dim=1)
        # min_distance_idx_1 = torch.clamp(min_distance_idx+1, min=0, max=self.translated_coordinates.size(1) - 1)
        # desired_heading = self.translated_coordinates[torch.arange(distances.shape[0]), min_distance_idx_1] - \
        #                   self.translated_coordinates[torch.arange(distances.shape[0]), min_distance_idx]
        #desired_heading_angle = torch.atan2(desired_heading[:,1], desired_heading[:,0])
        #heading_angle_Error = torch.abs(torch.abs(desired_heading_angle) - torch.abs(heading_angle_hunter))
        heading_angle_Error = heading_angle_hunter - self.cyaw_torch[min_distance_idx]
        heading_angle_Error = torch.abs((heading_angle_Error + torch.pi)%(2*torch.pi) - torch.pi)
        
        self.crosstrack_error = torch.abs(distances[torch.arange(distances.shape[0]), min_distance_idx])
        self.crosstrack_error -= 0.0
        self.crosstrack_error /= 5.0
        heading_angle_Error -= 0.0
        heading_angle_Error /= math.pi

        base_lin_vel = torch.abs(self.hunter.data.root_lin_vel_b[:,0])
        base_lin_vel -= 0.0
        base_lin_vel /= 3.0
        
        reward_ce = torch.exp(-1.0*self.crosstrack_error)
        reward_he = torch.exp(-1.0*heading_angle_Error) 
        reward_vel = 0.1*base_lin_vel
        reward_reset = -1.0*self.hunter_reset
        total_reward = reward_ce*reward_he*reward_vel + reward_reset
        total_reward = torch.clip(total_reward, min=0.0, max=None)
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        coordinates_hunter = self.hunter.data.root_pos_w[:,0:2].unsqueeze(1)    
        distances = torch.sqrt(torch.sum((coordinates_hunter - self.translated_coordinates) ** 2, dim=-1))
        min_distance_idx = torch.argmin(distances, dim=1)
        self.crosstrack_error2 = distances[torch.arange(distances.shape[0]), min_distance_idx]
        base_lin = self.hunter.data.root_lin_vel_b[:,0]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        hunter_reset_vel = torch.where(base_lin <= 0.01, 1.0, 0.0).bool()
        hunter_reset_crosstrack = torch.where(self.crosstrack_error2 >= 5.0, 1.0, 0.0).bool()
        self.hunter_reset = hunter_reset_vel | hunter_reset_crosstrack

        return self.hunter_reset, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self.hunter._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.hunter.data.default_joint_pos[env_ids]
        joint_vel = self.hunter.data.default_joint_vel[env_ids]

        default_root_state = self.hunter.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
       

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.hunter.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.hunter.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.hunter.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

           
