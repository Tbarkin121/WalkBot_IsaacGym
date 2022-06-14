"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import os
import math
from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
import torch
import time
import yaml

class WalkBot_testing():
    def __init__(self):
        with open("../../training/cfg/task/WalkBot.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        self.num_envs = 1
        self.create_sim()
        self.create_envs(self.num_envs, 1, 2)
        self.get_state_tensors()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Look at the first env
        cam_pos = gymapi.Vec3(2, 1, 1)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_I, "forward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_J, "left")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_K, "backward")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "right")


        self.simulation_loop()

    def create_sim(self):
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # parse arguments
        args = gymutil.parse_arguments(description="Joint control Methods Example")

        # create a simulator
        sim_params = gymapi.SimParams()
        sim_params.substeps = 2
        sim_params.dt = 1.0 / 100.0


        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 2

        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

        sim_params.use_gpu_pipeline = False
        if args.use_gpu_pipeline:
            print("WARNING: Forcing CPU pipeline.")
        self.device = 'cpu'

        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')

        
    def create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, 0.0, spacing)

        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.up_axis_idx = 2 #2 for z axis
        self.gym.add_ground(self.sim, plane_params)
        

        collision_group = 0
        collision_filter = 0


        # add cartpole urdf asset
        asset_root = "../../assets"
        asset_file = "urdf/WalkBot/urdf/WalkBot.urdf"
        # asset_file = "urdf/WalkBot_3DOF_330/urdf/WalkBot_3DOF.urdf"

        # Load asset with default control type of position for all joints
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 10000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
        walkbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # initial root pose for cartpole actors
        initial_pose = gymapi.Transform()
        initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)

        # Create environmentself.tensebot_handles = []
        self.walkbot_handles = []
        self.envs = []        
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            # Creates a tensegrity bot for an environment
            # Returns a list of handles for the support actors
            walkbot_handle = self.gym.create_actor(env_ptr, walkbot_asset, initial_pose, 'WalkBot', 0, 0)

            props = self.gym.get_actor_dof_properties(env_ptr, walkbot_handle)
            props["driveMode"][:] = gymapi.DOF_MODE_POS
            props["stiffness"] = 1000.0
            props['damping'][:] = 100.0
            props['velocity'][:] = 10.89
            props['effort'][:] = 0.52
            props['friction'][:] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, walkbot_handle, props)

            self.walkbot_handles.append(walkbot_handle)
            self.envs.append(env_ptr)

        self.num_actors = self.gym.get_actor_count(self.envs[0])
        self.num_bodies = self.gym.get_env_rigid_body_count(self.envs[0])
        # Set DOF drive targets
        dof_dict = self.gym.get_actor_dof_dict(env_ptr, walkbot_handle)
        joint_dict = self.gym.get_actor_joint_dict(env_ptr, walkbot_handle)
        dof_keys = list(dof_dict.keys())

    def get_state_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.pos = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.ori = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[..., 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.walkbot_init_pos = self.pos.clone()
        self.walkbot_init_ori = self.ori.clone()
        
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(torch.squeeze(self.walkbot_init_ori)).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()


    def simulation_loop(self):
        # Simulate
        joint_idx = 0
        control_idx = 0
        loop_counter = 1
        max_loops = 50

        while not self.gym.query_viewer_has_closed(self.viewer):
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "forward" and evt.value > 0:
                    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    forces[:, 0, 0] = torch.tensor([1], device=self.device)
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

                elif (evt.action == "backward") and evt.value > 0:
                    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    forces[:, 0, 0] = torch.tensor([-1], device=self.device)
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)                

                elif (evt.action == "left") and evt.value > 0:
                    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    forces[:, 0, 1] = torch.tensor([1], device=self.device)
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

                elif (evt.action == "right") and evt.value > 0:
                    forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
                    forces[:, 0, 1] = torch.tensor([-1], device=self.device)
                    self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)

            
            self.gym.refresh_actor_root_state_tensor(self.sim)

            
            goal_pos = torch.tensor([1,0,0], device=self.device).repeat(self.num_envs,1)

            obs = test_jit_script(torch.squeeze(self.pos, dim=1),
                                  torch.squeeze(self.ori, dim=1),
                                  torch.squeeze(self.linvel, dim=1),
                                  torch.squeeze(self.angvel, dim=1),
                                  goal_pos,
                                  self.inv_start_rot,
                                  self.basis_vec0,
                                  self.basis_vec1)
            os.system('cls||clear')
            print('World LinVel = {}'.format(obs[0,:]))
            print('Local LinVel = {}'.format(obs[1,:]))
            print('World AngVel = {}'.format(obs[2,:]))
            print('Local AngVel = {}'.format(obs[3,:]))
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            # if(loop_counter == 0):
            #     print('control idx = {}. handle_list[{}] = {}'.format(control_idx, joint_idx, joint_idx))
            #     if(control_idx == 0):
            #         self.gym.set_dof_target_position(self.envs[0], joint_idx, 2.09)
            #     elif(control_idx == 1):
            #         self.gym.set_dof_target_position(self.envs[0], joint_idx, -2.09)
            #     else:
            #         self.gym.set_dof_target_position(self.envs[0], joint_idx, 0)
            #     control_idx += 1
            #     if(control_idx>2):
            #         control_idx = 0
            #         joint_idx += 1
            #         if(joint_idx > 5):
            #             joint_idx = 0

            # loop_counter += 1
            # if(loop_counter > max_loops):
            #     loop_counter=0

        
            # Wait for dt to elapse in real time.
            # This synchronizes the physics simulation with the rendering rate.
            self.gym.sync_frame_time(self.sim)

        print('Done')

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def test_jit_script(
    walkbot_pos,
    walkbot_ori,
    walkbot_linvel,
    walkbot_angvel,
    goal_pos,
    inv_start_rot,
    basis_vec0,
    basis_vec1
    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    
    to_target = goal_pos - walkbot_pos
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        walkbot_ori, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, walkbot_linvel, walkbot_angvel, goal_pos, walkbot_pos)
    obs = torch.cat((walkbot_linvel, vel_loc, walkbot_angvel, angvel_loc))
    return obs


test = WalkBot_testing()