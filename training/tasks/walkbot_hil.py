from turtle import update
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

from tasks import walkbot_hardware

class WalkBot_HIL(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.hardware = walkbot_hardware.WalkBot_Hard()
        # self.hardware.enable_torque()
        self.hardware.disable_torque()



        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]
        self.dt_mod = self.cfg["sim"]["dt_mod"]

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]*self.dt_mod
        # self.randomization_params = self.cfg["task"]["randomization_params"]
        # self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        # self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        # self.power_scale = self.cfg["env"]["powerScale"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_mode = self.cfg["env"]["debug_mode"]
        self.her_mode = self.cfg["env"]["her_obs"]
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

        self.goal_offset = self.cfg["env"]["goalOffset"]
        self.goal_dist = self.cfg["env"]["goalDist"]
        self.goal_threshold = self.cfg["env"]["goalThreshold"]
        self.goal_range = self.cfg["env"]["goalRange"]

        # obs_buf shapes: 12
        # DOF POS (3)
        # DOF VEL (3)
        # Goal POS (3)
        # Actions (3)
        self.cfg["env"]["numObservations"] = 12
        # Joint Targets (3) Either Torques or Positions, depends on drive mode 
        self.cfg["env"]["numActions"] = 3

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU DOF state tensor
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # get gym root state tensor
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.servo_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 1:4, 0:3]
        self.servo_ori = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 1:4, 3:7]


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
        self.walkbot_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, :]
        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 1, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.goal_vel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 1, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        
        self.walkbot_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0, :]
        self.walkbot_initial_root_states = self.walkbot_root_state.clone()
        self.walkbot_initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

        # Used for rewarding moving towards a target
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.goal_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        goal_ids = self.goal_reset.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_ids) > 0:
            self.reset_goal(goal_ids)

        self.frame_count = 0
        # self.plot_buffer = []
        self.prev_dof_scaled_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_dof_scaled_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_scaled_goal_position = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, 3), device=self.device)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "enable")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "disable")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "debug")

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
        asset_file = "urdf/IK_RL/urdf/WalkBot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.angular_damping = self.angularDamping
        asset_options.max_angular_velocity = self.angularVelocity

        walkbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(walkbot_asset)

        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.01, asset_options)
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
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal", -1, 1, 0)
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.2, 0.8, 0.2))
            self.goal_handles.append(goal_handle)


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

        # Modify ENV 0 to be controlled with hardware
        dof_props = self.gym.get_actor_dof_properties(self.envs[0], self.walkbot_handles[0])
        dof_props['driveMode'][:] = gymapi.DOF_MODE_NONE
        dof_props['stiffness'][:] = 0.0
        dof_props['damping'][:] = 0.0
        # dof_props['friction'][:] = 10000
        self.gym.set_actor_dof_properties(self.envs[0], walkbot_handle, dof_props)


    def find_foot_coordinates(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        foot_offset = torch.tensor([0,0,-0.048], device=self.device) # Distance from servo2 to the end of the foot
        # Step 1, Construct Rotation Matrix from Quaternions
        # Just doing this to follow the notation from the Quaternion to Rotation Matrix math
        q0 = self.servo_ori[:,2,3]
        q1 = self.servo_ori[:,2,0]
        q2 = self.servo_ori[:,2,1]
        q3 = self.servo_ori[:,2,2]
        
        Rot_Mats = torch.zeros((self.num_envs, 3, 3), device=self.device)
        Rot_Mats[:,0,0] = 2*(q0**2 + q1**2)-1
        Rot_Mats[:,1,0] = 2*(q1*q2 + q0*q3)
        Rot_Mats[:,2,0] = 2*(q1*q3 - q0*q2)
        Rot_Mats[:,0,1] = 2*(q1*q2 - q0*q3)
        Rot_Mats[:,1,1] = 2*(q0**2 + q2**2)-1
        Rot_Mats[:,2,1] = 2*(q2*q3 + q0*q1)
        Rot_Mats[:,0,2] = 2*(q1*q3 + q0*q2)
        Rot_Mats[:,1,2] = 2*(q2*q3 - q0*q1)
        Rot_Mats[:,2,2] = 2*(q0**2 + q3**2)-1

        # Step 2, rotate the origional foot offset for each environment
        self.foot_pos = torch.matmul(Rot_Mats, foot_offset)
        # Step 3, add the servo position to the results from step 2
        self.foot_pos += self.servo_pos[:,2,:]
        

    def compute_reward(self):
        # to_target = self.goal_pos - self.foot_pos
        # distance_to_goal = torch.norm(to_target, dim=-1)
        # reward = -distance_to_goal*2
        # print('D2G, R = {}, {}'.format(distance_to_goal, reward))

        self.rew_buf[:], self.reset_buf[:], self.goal_reset, self.reward_comp= compute_walkbot_reward(
            self.obs_buf,
            self.foot_pos,
            self.goal_pos,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.max_episode_length,
            self.goal_threshold,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale)
        # self.rew_buf = self.rew_buf/self.dt_mod
        # to_target = self.foot_pos - self.goal_pos
        # distance_to_goal = torch.norm(to_target, p=2, dim=-1)
        # distance_reward = (1.0/(1.0+distance_to_goal**2))**2
        # distance_reward = torch.where(distance_to_goal <= 0.02, distance_reward * 2, distance_reward )
        # print(distance_to_goal)

    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)


        goalOffset = torch.tensor(self.goal_offset, device=self.device)
        
        dof_scaled_position = torch.div(self.dof_pos, self.maxPosition)
        dof_scaled_velocity = torch.div(self.dof_vel, self.maxSpeed)
        scaled_goal_position = torch.div(self.goal_pos-goalOffset, self.goal_range)
        scaled_foot_position = torch.div(self.foot_pos-goalOffset, self.goal_range) #The coordinates of this should be identical to that of the scaled goal position
        
        # HER modification for half of the environments
        if(self.her_mode):
            tmp_incr = torch.unsqueeze(torch.range(0, self.num_envs-1, device=self.device), dim=-1).repeat(1,3)
            scaled_goal_position = torch.where(tmp_incr < len(tmp_incr)/2 , scaled_goal_position, scaled_foot_position)        

        self.obs_buf = torch.cat((dof_scaled_position, 
                                  dof_scaled_velocity, 
                                  scaled_goal_position,
                                  self.actions), dim=-1)
        # self.obs_buf = torch.cat((dof_scaled_position, 
        #                           dof_scaled_velocity, 
        #                           scaled_goal_position, 
        #                           self.actions,
        #                           self.prev_dof_scaled_position,
        #                           self.prev_dof_scaled_velocity,
        #                           self.prev_scaled_goal_position,
        #                           self.prev_actions), dim=-1)
        # self.prev_dof_scaled_position = dof_scaled_position
        # self.prev_dof_scaled_velocity = dof_scaled_velocity
        # self.prev_scaled_goal_position = scaled_goal_position
        # self.prev_actions = self.actions
        # self.obs_buf[:] = compute_walkbot_observations(
        #     self.dof_pos,
        #     self.dof_vel,
        #     self.goal_pos, 
        #     self.actions,
        #     goalOffset+goalOffset,
        #     self.maxPosition,
        #     self.maxSpeed
        # )
            
        # # print(self.obs_buf[:,0])
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


    def reset_goal(self, env_ids):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('Resetting Goals! Env_IDs = {}'.format(env_ids))
        # print('Old Goal Position = {}'.format(self.goal_pos))
        # print(self.goal_offset)
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        goal_pos_update = torch_rand_float(-self.goal_dist, self.goal_dist, (len(env_ids), 3), device=self.device)
        self.goal_pos[env_ids, :] = goal_pos_update + torch.tensor(self.goal_offset, device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32+1), len(env_ids_int32))

        self.goal_reset[env_ids] = 0
        # time.sleep(0.05)
        # print('New Goal Position = {}'.format(self.goal_pos))
    
    def kick_goal(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        tmp = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        env_ids = tmp.nonzero(as_tuple=False).squeeze(-1)
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        
        # update_roll = torch.squeeze(torch_rand_float(0.0, 1.0, (self.num_envs, 1), device=self.device))
        update_roll = torch_rand_float(0.0, 1.0, (self.num_envs, 1), device=self.device)
        ball_home = self.servo_pos[0,0,:]
        vec_to_home = self.goal_pos - ball_home
        dist_to_home = torch.unsqueeze(torch.norm(vec_to_home, dim=1), -1)
        dir_to_home = torch.div(vec_to_home, dist_to_home.repeat(1,3))
       
        goal_vel_update = torch_rand_float(-0.05, 0.05, (len(env_ids), 3), device=self.device)
        # Give the goal a new random velocity 5% of the time
        goal_vel_update = torch.where(update_roll<0.05, goal_vel_update, self.goal_vel)
        # Prevent the goal from getting too far away
        goal_vel_update = torch.where(dist_to_home > self.goal_range, -dir_to_home*.1, goal_vel_update)
        # Prevent the goal from being too low
        goal_height= torch.unsqueeze(self.goal_pos[:,2],dim=-1)
        root_height = torch.unsqueeze(self.walkbot_initial_root_states[:,2],dim=-1)
        goal_vel_update = torch.where(goal_height < root_height, -dir_to_home*.1, goal_vel_update)
        # print(goal_vel_update)
        # time.sleep(0.1)
        self.goal_vel[env_ids, :] = goal_vel_update
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.root_states),
                                              gymtorch.unwrap_tensor(env_ids_int32+1), len(env_ids_int32))

        self.goal_reset[env_ids] = 0

        # env_ids_int32 = goal_ids.to(dtype=torch.int32)*self.num_actors

        # goal_vel_update = torch_rand_float(-1, 1, (len(goal_ids), 3), device=self.device)
        # # self.goal_vel = torch.where(update_roll < 0.05, goal_vel_update, torch.zeros_like(goal_vel_update))
        # self.goal_pos = goal_vel_update
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.root_states),
        #                                       gymtorch.unwrap_tensor(env_ids_int32+1), len(env_ids_int32))

    def pre_physics_step(self, actions):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "enable" and evt.value > 0:
                self.hardware.enable_torque()
            elif (evt.action == "disable") and evt.value > 0:
                self.hardware.disable_torque()
            elif (evt.action == "debug") and evt.value > 0:
                self.debug_mode = not self.debug_mode

        self.find_foot_coordinates()
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        # self.actions = torch.ones_like(self.actions)
        # print(self.actions)
        if(self.drive_mode):
            # Pos Control
            self.set_motor_positions(self.actions)
        else:
            # Torque Control
            self.set_motor_torques(self.actions)
        
        # Send actions to servos
        pos_list = [self.actions[0,0], self.actions[0,1], self.actions[0,2]]
        self.hardware.write_positions(walkbot_hardware.DXL_IDS, pos_list)
        self.kick_goal()

        # Hardware will control ENV 0 
        state = torch.tensor(self.hardware.read_state(), device=self.device)
        env_ids = torch.tensor([0], device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)*self.num_actors
        positions = torch.tensor([state[0,0],state[0,1],state[0,2]], device=self.device, dtype=torch.float)
        velocities = torch.tensor([state[1,0],state[1,1],state[1,2]], device=self.device, dtype=torch.float)
        self.dof_pos[env_ids, :] = positions[:] * self.maxPosition
        self.dof_vel[env_ids, :] = velocities[:] * self.maxSpeed

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        
    def set_motor_positions(self, targets):
        target_pos = targets*self.maxPosition
        # print(target_pos)
        # target_pos = torch.ones_like(targets, device=self.device)*self.maxPosition
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target_pos))
    
    def set_motor_torques(self, targets):
        # print('torques?')
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
        env_idx = 0
        camOffset = gymapi.Vec3(0, -0.5, 0.25)
        camTarget = gymapi.Vec3(self.goal_pos[env_idx, 0],self.goal_pos[env_idx, 1],self.goal_pos[env_idx, 2])
        camEnvOffset = gymapi.Vec3(0, 0, 0)
        # print(camOffset)
        # print(camTarget)
        # self.gym.viewer_camera_look_at(self.viewer, None, camOffset+camTarget+camEnvOffset, camTarget+camEnvOffset)
        # time.sleep(0.1)
        if(self.debug_mode):
            self.debug_printout()

    def debug_printout(self):
        print('DEBUG PRINTOUTS')
        # print('target pos = {}'.format(self.actions*self.maxPosition))
        print(self.obs_buf[0, : ])
        # print(self.rew_buf)
        # print(self.actions)
        # print(self.reward_comp[0,:])
        # print('goal pos : {}'.format(self.obs_buf[:, 6:9]))
        # time.sleep(.1)
        # draw some lines
        self.gym.clear_lines(self.viewer)
        num_lines = 1
        line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
        line_color = torch.tensor((0.0, 1.0, 1.0), device=self.device, dtype=torch.float)
        line_vertices[0,:] = self.goal_pos[0,:]
        line_vertices[1,:] = self.foot_pos[0,:]
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())
        
        num_lines = 1
        line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
        line_color = torch.tensor((1.0, 1.0, 0.0), device=self.device, dtype=torch.float)
        line_vertices[0,:] = self.goal_pos[0,:]
        line_vertices[1,:] = self.servo_pos[0,0,:]
        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())

        # print(self.rew_buf[0])
        # print(self.actions[0,:])
        # for rew in self.reward_comp:
            # print(rew)
        # print(self.actions[0,:])
        # print(self.reward_comp[1])
        # norm_vel = torch.norm(self.dof_vel, dim=-1)
        # print(norm_vel)
        # print(self.actions[0,...])
        

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_walkbot_reward(
    obs_buf,
    foot_pos,
    goal_pos,
    reset_buf,
    progress_buf,
    actions,
    max_episode_length,
    goal_threshold,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    
    to_target = foot_pos - goal_pos
    distance_to_goal = torch.norm(to_target, p=2, dim=-1)
    distance_reward = (1.0/(1.0+distance_to_goal**2))**2
    distance_reward = torch.where(distance_to_goal <= 0.02, distance_reward * 2, distance_reward )
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
    
    # norm_vel = torch.norm(dof_vel/max_dov_vel, dim=-1)
    # reward -= norm_vel
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 3:6]), dim=-1)
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 0:3]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 0:3]) > 0.98) * scaled_cost, dim=-1)
    near_goal_reward = torch.where(distance_to_goal < goal_threshold, 1, 0)
    velocity_cost = torch.sum(torch.abs(obs_buf[:, 3:6]), dim=-1)/3
    # goal_reached = torch.where(distance_to_goal < goal_threshold, 1, 0)
    
    rewards = torch.zeros_like(distance_reward)
    rewards += distance_reward
    # rewards -= actions_cost_scale*actions_cost            # Action Costs don't make sense in Position Control Mode
    # rewards -= electricity_cost*energy_cost_scale         # Electrical Costs aren't calculated correctly in Position Control Mode
    rewards -= dof_at_limit_cost
    rewards -= velocity_cost**2
    # rewards += near_goal_reward
    # reward = torch.where(distance_to_goal < goal_threshold, reward+1, reward)
    # goal_reset = torch.where(goal_reached==1, 1, 0)
    # goal_reset = torch.where(progress_buf%(max_episode_length/10) == 0, 1, 0)
    goal_reset = torch.zeros_like(reset)
    reward_comps = torch.cat((torch.unsqueeze(rewards,dim=-1), 
                             torch.unsqueeze(distance_reward,dim=-1),
                             torch.unsqueeze(-actions_cost*actions_cost_scale,dim=-1), 
                             torch.unsqueeze(-electricity_cost*energy_cost_scale,dim=-1), 
                             torch.unsqueeze(-dof_at_limit_cost,dim=-1),
                             torch.unsqueeze(-velocity_cost,dim=-1)), dim=-1)

    return rewards, reset, goal_reset, reward_comps


@torch.jit.script
def compute_walkbot_observations(dof_pos,                       #Tensor
                                 dof_vel,                       #Tensor
                                 goal_pos,                      #Tensor
                                 actions,                       #Tensor
                                 maxGoalPos,                    #Tensor
                                 maxPosition,                   #float
                                 maxSpeed,                      #float
                                 ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dof_scaled_position = dof_pos / maxPosition
    dof_scaled_velocity = dof_vel / maxSpeed
    scaled_goal_position = goal_pos / maxGoalPos
        
    obs = torch.cat((dof_scaled_position, dof_scaled_velocity, scaled_goal_position, actions), dim=-1)

    return obs