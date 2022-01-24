import numpy as np
import os
import torch
import time
import matplotlib.pyplot as plt

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask


class WalkBot(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        # self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        # self.energy_cost_scale = self.cfg["env"]["energyCost"]
        # self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        # self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.drive_mode = self.cfg["env"]["actuatorParams"]["driveMode"]
        self.stiffness = self.cfg["env"]["actuatorParams"]["stiffness"] * self.drive_mode
        self.damping = self.cfg["env"]["actuatorParams"]["damping"] * self.drive_mode
        self.maxPosition = self.cfg["env"]["actuatorParams"]["maxPosition"]
        self.maxSpeed = self.cfg["env"]["actuatorParams"]["maxSpeed"]
        self.maxTorque = self.cfg["env"]["actuatorParams"]["maxTorque"]
        self.friction = self.cfg["env"]["actuatorParams"]["friction"]
        self.torqueDecay = self.cfg["env"]["actuatorParams"]["torqueDecay"]

        self.angularDamping = self.cfg["env"]["assetParams"]["angularDamping"]
        self.angularVelocity = self.cfg["env"]["assetParams"]["angularVelocity"]

        self.goal_dist = self.cfg["env"]["goalDist"]
        self.goal_threshold = self.cfg["env"]["goalThreshold"]

        # obs_buf shapes: (37)
        # obs_buf[0:13] = Root State : Pos(3), Ori(4), LinVel(3), AngVel(3)
        # obs_buf[13:25] = Dof State : Pos(6), Vel(6) 
        # obs_buf[25:28] = Goal Pos : Pos(3)
        # obs_buf[28] = angle_to_target : Projection(1)
        # obs_buf[29] = up_proj : Projection(1)
        # obs_buf[30] = heading_proj : Projection(1)
        # obs_buf[31:37] = actions : TargetTorques(6) or TargetPos(6) depends on drive mode
        self.cfg["env"]["numObservations"] = 37
        # Joint Targets (6) Either Torques or Positions, depends on drive mode 
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU DOF state tensor
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # set init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = torch.tensor(state, device=self.device)
        self.start_rotation = torch.tensor(rot, device=self.device)
        
        # get gym GPU root state tensor
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.walkbot_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.walkbot_ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.walkbot_linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.walkbot_angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 1, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        self.walkbot_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, :]
        self.walkbot_initial_root_states = self.walkbot_root_state.clone()
        self.walkbot_initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

        # Used for rewarding moving towards a target
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        to_target = self.goal_pos - self.walkbot_pos
        to_target[:, 2] = 0.0
        self.potentials = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.prev_potentials = self.potentials.clone()
        
        self.goal_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        # Measurements for rewards
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        
        self.frame_count = 0
        self.plot_buffer = []

    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(0.5 * -spacing, -spacing, 0.0)
        upper = gymapi.Vec3(0.5 * spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/WalkBot/urdf/WalkBot.urdf"
        # asset_file = "urdf/cassie/urdf/cassie.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = self.angularDamping
        asset_options.max_angular_velocity = self.angularVelocity

        walkbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(walkbot_asset)

        goal_asset = self.gym.create_sphere(self.sim, 0.05)
        self.num_bodies = self.gym.get_asset_rigid_body_count(walkbot_asset) + self.gym.get_asset_rigid_body_count(goal_asset)
        # self.num_actor = get_sim_actor_count
        
        pose = gymapi.Transform()

        self.walkbot_handles = []
        self.goal_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            walkbot_handle = self.gym.create_actor(env_ptr, walkbot_asset, pose, "walkbot", i, 1, 0)
            rand_color = torch.rand((3), device=self.device)
            for j in range(self.num_bodies):
                # self.gym.set_rigid_body_color(
                #     env_ptr, walkbot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.27, 0.1, 0.66))
                self.gym.set_rigid_body_color(
                    env_ptr, walkbot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
                

            dof_props = self.gym.get_actor_dof_properties(env_ptr, walkbot_handle)
            if(self.drive_mode):
                dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            else:
                dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = self.stiffness
            dof_props['damping'][:] = self.damping
            dof_props['velocity'][:] = self.maxSpeed
            dof_props['effort'][:] = self.maxTorque
            dof_props['friction'][:] = self.friction

            self.gym.set_actor_dof_properties(env_ptr, walkbot_handle, dof_props)

            self.envs.append(env_ptr)
            self.walkbot_handles.append(walkbot_handle)

            # Set Up the Goal Actor
            goal_pose = gymapi.Transform()
            goal_pose.p.y = self.goal_dist
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", i, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.goal_handles.append(goal_handle)


        for j in range(self.num_dof):
            if dof_props['lower'][j] > dof_props['upper'][j]:
                self.dof_limits_lower.append(dof_props['upper'][j])
                self.dof_limits_upper.append(dof_props['lower'][j])
            else:
                self.dof_limits_lower.append(dof_props['lower'][j])
                self.dof_limits_upper.append(dof_props['upper'][j])
        
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, walkbot_handle)
        self.joint_dict = self.gym.get_actor_joint_dict(env_ptr, walkbot_handle)
        self.dof_dict = self.gym.get_asset_dof_dict(walkbot_asset)
        print('body_dict:')
        print(self.body_dict)
        for b in self.body_dict:
                print(b)
        print('joint_dict:')
        for j in self.joint_dict:
            print(j)
        print('dof_dict:')
        for d in self.dof_dict:
            print(d)
        
                
    def compute_reward(self):
        # box_pos = self.obs_buf[:, 0:3]
        # # print(box_pos[:, 2])
        # # print(box_pos.shape)
        
        # # box_ori = self.obs_buf[:, 3:7]
        # # box_lin_vel = self.obs_buf[:, 7:10]
        # # box_ang_vel = self.obs_buf[:, 10:13]
        # # print(self.corner1_pos)
        # # print(self.corner1_pos.shape)
        # self.rew_buf[:], self.reset_buf[:] = compute_walkbot_reward(
        #     box_pos, self.reset_buf, self.progress_buf, self.max_episode_length
        # )
        # # print(self.rew_buf)
        # # print(self.rew_buf.shape)
        # # print(self.reset_buf)
        # # print(self.reset_buf.shape)
        self.rew_buf[:], self.reset_buf[:], self.goal_reset = compute_walkbot_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.goal_threshold)
        
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print('self.root_state')
        # print(self.root_states[0,:])
        # print(self.root_states.shape)
        # time.sleep(1)
                
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_walkbot_observations(
            self.walkbot_pos,
            self.walkbot_ori,
            self.walkbot_linvel,
            self.walkbot_angvel,
            self.dof_pos,
            self.dof_vel,
            self.goal_pos, 
            self.potentials,
            self.inv_start_rot, 
            self.dof_limits_lower, 
            self.dof_limits_upper, 
            self.dof_vel_scale,
            self.actions, 
            self.dt,
            self.basis_vec0, 
            self.basis_vec1, 
            self.up_axis_idx,
            self.maxSpeed)
        # print(self.obs_buf[:,0])
        return self.obs_buf

    def reset_idx(self, env_ids):
        # print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors

        self.walkbot_root_state[env_ids, :] = self.walkbot_initial_root_states[env_ids, :]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


        
        positions = 0.0 * (torch.rand((len(env_ids), self.num_dof), device=self.device))
        velocities = 0.0 * (torch.rand((len(env_ids), self.num_dof), device=self.device))
        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.goal_reset[env_ids] = 1
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)


        # plt.plot([0,0,0])
        # plt.show()
        # if(self.plot_buffer):
        #     plot_data = np.array(self.plot_buffer)
        #     print(plot_data.shape)
        #     plt.plot(plot_data[:,0,0] + plot_data[:,1,0] + plot_data[:,2,0], label="Total Reward")
        #     plt.plot(plot_data[:,0,0], label="Progress Reward")
        #     plt.plot(plot_data[:,1,0], label="Height Reward")
        #     plt.plot(plot_data[:,2,0], label="Heading Reward")
        #     plt.ylabel('Reward')
        #     plt.xlabel('Steps')
        #     plt.grid()
        #     plt.legend(loc="lower right")
        #     plt.xlim([0, 500])
        #     plt.ylim([-0.1, 2.1])
        #     plt.show()
        #     self.plot_buffer = []

    def reset_goal(self, env_ids):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('Resetting Goals! Env_IDs = {}'.format(env_ids))
        # print('Old Goal Position = {}'.format(self.goal_pos))

        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        goal_pos_update = torch_rand_float(-self.goal_dist, self.goal_dist, (len(env_ids), 3), device=self.device)
        # goal_pos_update[:,0] = 1000.0
        # goal_pos_update[:,1] = 0.0
        goal_pos_update[:,2] = 0.1
        self.goal_pos[env_ids, :] = goal_pos_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32+1), len(env_ids_int32))

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        to_target = self.goal_pos[env_ids] - self.walkbot_pos[env_ids]
        to_target[:, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.goal_reset[env_ids] = 0
        # print('New Goal Position = {}'.format(self.goal_pos))

    def pre_physics_step(self, actions):
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        if(self.drive_mode):
            # Pos Control
            self.set_motor_positions(self.actions)
        else:
            # Torque Control
            self.set_motor_torques(self.actions)


        # # Apply random forces (Turn this into a useful function instead of floating code...)
        # if (self.frame_count - 99) % 200 == 0:
        #     # set forces and torques for the ant root bodies
        #     forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        #     torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        #     forces[:, 0, 0:3] = torch.distributions.Uniform(-2, 2).sample(torch.Size([3]))
        #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        # self.frame_count += 1

    def set_motor_positions(self, targets):
        target_pos = targets*self.maxPosition
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_pos))
    
    def set_motor_torques(self, targets):
        # target_torques = torch.zeros((self.num_envs, self.num_dof), device=self.device)

        target_torques = targets*self.maxTorque

        # print('target_torques = {}'.format(target_torques))

        # Clean this up later
        offset = self.torqueDecay
        max_available_torque = torch.clip(self.maxTorque - (offset*self.dof_vel/self.maxSpeed + (1-offset))*self.maxTorque, -self.maxTorque, self.maxTorque)
        min_available_torque = torch.clip(-self.maxTorque - (offset*self.dof_vel/self.maxSpeed - (1-offset))*self.maxTorque, -self.maxTorque, self.maxTorque)
        self.torques = torch.clip(target_torques, min_available_torque, max_available_torque)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        self.compute_observations()
        self.compute_reward()

        # Look at the first actor
        env_idx = 3
        camOffset = gymapi.Vec3(0, -1.5, 0.25)
        camTarget = gymapi.Vec3(self.walkbot_pos[env_idx, 0],self.walkbot_pos[env_idx, 1],self.walkbot_pos[env_idx, 2])
        camEnvOffset = gymapi.Vec3(1, 1, 0)
        # print(camOffset)
        # print(camTarget)
        self.gym.viewer_camera_look_at(self.viewer, None, camOffset+camTarget+camEnvOffset, camTarget+camEnvOffset)
        # time.sleep(0.1)   
        self.debug_printout()

    def debug_printout(self):
        # print('DEBUG PRINTOUTS')
        # body_height = self.obs_buf[:,2]
        # up_projection = self.obs_buf[:,29]
        # heading_projection = self.obs_buf[:, 30] 
        # heading_reward = self.heading_weight * heading_projection    
        # # aligning up axis and environment
        # up_reward = torch.zeros_like(heading_reward)
        # up_reward = torch.where(up_projection > 0.93, up_reward + self.up_weight, up_reward)
        # # reward for duration of staying alive
        # progress_reward = self.potentials - self.prev_potentials
        # total_reward = progress_reward + up_reward + heading_reward]
        xtream_rewards = torch.abs(self.rew_buf) > 5
        # print('ProgressReward[3] : {} = {} - {}'.format(progress_reward[3], self.potentials[3], self.prev_potentials[3]))
        # print('EnvReset[3], GoalReset[3] : {}, {}'.format(self.reset_buf[3], self.goal_reset[3]))
        # print('Bot Pos, Goal Pos = {}, {}'.format(self.walkbot_pos[3,:], self.goal_pos[3,:]))
        if(torch.any(xtream_rewards)):
            print('XTREAM REWARD DETECTED')
            xtream_idx = xtream_rewards.nonzero().cpu().detach().numpy()
            print("xtream index = {}".format(xtream_idx))
            print(self.rew_buf[xtream_idx])
            print('Progress Reward : {} = {} - {}'.format(progress_reward[xtream_idx], self.potentials[xtream_idx], self.prev_potentials[xtream_idx]))
            print('EnvReset, GoalReset : {},{}'.format(self.reset_buf[xtream_idx], self.goal_reset[xtream_idx]))
            time.sleep(10)
            print()
        # print('{:.2f} = {:.2f} + {:.2f} + {:.2f}'.format(total_reward[0], heading_reward[0], up_reward[0], progress_reward[0]))

        # print(' self.reset_buf')
        # print( self.reset_buf)
            # tmp_progress_reward = self.potentials - self.prev_potentials
            # if( np.abs(tmp_progress_reward[0].cpu().detach().numpy()) > 1):
            #     print('{} : {} : {}'.format(tmp_progress_reward[0], self.potentials[0], self.prev_potentials[0]))
            #     time.sleep(1)
        # tmp_height_reward = self.obs_buf[:,0]
        # tmp_heading_reward = self.rew_buf - tmp_progress_reward
        # self.plot_buffer.append((tmp_progress_reward.cpu().detach().numpy(),
        #                         tmp_height_reward.cpu().detach().numpy(),
        #                         tmp_heading_reward.cpu().detach().numpy()))
        

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_walkbot_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    termination_height,
    death_cost,
    max_episode_length,
    goal_threshold):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor]

    # reward from direction headed
    # heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    # heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)
    
    # obs_buf shapes: 3, 4, 3, 3, 6, 6, 3, 1, 1, 1, 6 = 37
    # walkbot_pos                           0,1,2
    # walkbot_ori                           3,4,5,6
    # vel_loc                               7,8,9
    # angvel_loc                            10,11,12
    # dof_pos_scaled                        13,14,15,16,17,18
    # dof_vel_scaled                        19,20,21,22,23,24
    # goal_pos                              25,26,27
    # angle_to_target.unsqueeze(-1)         28
    # up_proj.unsqueeze(-1)                 29
    # heading_proj.unsqueeze(-1)            30
    # actions                               31,32,33,34,35,36
                     
    # Old Obs (30): 
    # body_height = obs_buf[:,0]
    # up_projection = obs_buf[:,10]
    # heading_projection = obs_buf[:, 11] 
    # New Obs (37): 
    body_height = obs_buf[:,2]
    up_projection = obs_buf[:,29]
    heading_projection = obs_buf[:, 30] 
    left_foot_pos = obs_buf[:,15]
    right_foot_pos = obs_buf[:,18]

    heading_reward = heading_weight * heading_projection    
    # aligning up axis and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_projection > 0.93, up_reward + up_weight, up_reward)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    foot_diff_penelty = torch.abs(left_foot_pos-right_foot_pos)/10
    
    total_reward = progress_reward + up_reward + heading_reward - foot_diff_penelty
    # total_reward = progress_reward + obs_buf[:,0]*4 + heading_reward
    # total_reward = progress_reward + heading_reward
    # total_reward = progress_reward + obs_buf[:,0]
    # total_reward = obs_buf[:,0]

    # adjust reward for fallen agents
    total_reward = torch.where(body_height < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(body_height < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    walkbot_pos = obs_buf[:,0:3]
    goal_pos = obs_buf[:,25:28]
    distance_to_goal = torch.norm(walkbot_pos - goal_pos, dim=-1)
    goal_reached = torch.where(distance_to_goal < goal_threshold, 1, 0)
    goal_reset = torch.where(goal_reached==1, 1, 0)

    return total_reward, reset, goal_reset

@torch.jit.script
def compute_walkbot_observations(walkbot_pos,                   #Tensor
                                 walkbot_ori,                   #Tensor
                                 walkbot_linvel,                #Tensor
                                 walkbot_angvel,                #Tensor
                                 dof_pos,                       #Tensor
                                 dof_vel,                       #Tensor
                                 goal_pos,                      #Tensor
                                 potentials,                    #Tensor
                                 inv_start_rot,                 #Tensor
                                 dof_limits_lower,              #Tensor
                                 dof_limits_upper,              #Tensor
                                 dof_vel_scale,                 #float       
                                 actions,                       #Tensor
                                 dt,                            #float
                                 basis_vec0,                    #Tensor
                                 basis_vec1,                    #Tensor
                                 up_axis_idx,                   #int
                                 maxSpeed                       #float
                                 ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, int, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    to_target = goal_pos - walkbot_pos
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        walkbot_ori, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, walkbot_linvel, walkbot_angvel, goal_pos, walkbot_pos)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)
    dof_vel_scaled = dof_vel[:, 0:6]/maxSpeed
    # # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(6), num_dofs(6), num_dofs(6)
    # obs = torch.cat((walkbot_pos[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
    #                  yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
    #                  up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
    #                  dof_vel * dof_vel_scale,
    #                  actions), dim=-1)

    # obs_buf shapes: 3, 4, 3, 3, 6, 6, 3, 1, 1, 1, 6 = 37
    obs = torch.cat((walkbot_pos, walkbot_ori, vel_loc, angvel_loc, dof_pos_scaled, dof_vel_scaled, goal_pos, 
                     angle_to_target.unsqueeze(-1), up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec