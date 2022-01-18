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

import math
from isaacgym import gymapi
from isaacgym import gymutil
import time

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 60.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, gymapi.PlaneParams())

# set up the env grid
num_envs = 4
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

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
cubebot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 2.0, 0.0)
initial_pose.r = gymapi.Quat(0, 0.0, 0.0, 1.0)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
props["driveMode"][:] = gymapi.DOF_MODE_POS
props["stiffness"] = 100000
props['damping'][:] = 0.0
props['velocity'][:] = 10.89
props['effort'][:] = 0.52
props['friction'][:] = 0.0

gym.set_actor_dof_properties(env0, cubebot0, props)
# Set DOF drive targets
dof_dict = gym.get_actor_dof_dict(env0, cubebot0)
joint_dict = gym.get_actor_joint_dict(env0, cubebot0)
dof_keys = list(dof_dict.keys())

# targets = torch.tensor([1000, 0, 0, 0, 0, 0])
# gym.set_dof_velocity_target_tensor(env0, gymtorch.unwrap_tensor(targets))

# Look at the first env
cam_pos = gymapi.Vec3(8, 4, 1.5)
cam_target = gymapi.Vec3(0, 2, 1.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
joint_idx = 0
control_idx = 0
loop_counter = 1
max_loops = 50
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    if(loop_counter == 0):
        print('control idx = {}. handle_list[{}] = {}'.format(control_idx, joint_idx, joint_idx))
        if(control_idx == 0):
            gym.set_dof_target_position(env0, joint_idx, 1.5707)
        elif(control_idx == 1):
            gym.set_dof_target_position(env0, joint_idx, -1.5707)
        else:
            gym.set_dof_target_position(env0, joint_idx, 0)
        control_idx += 1
        if(control_idx>2):
            control_idx = 0
            joint_idx += 1
            if(joint_idx > 5):
                joint_idx = 0

    loop_counter += 1
    if(loop_counter > max_loops):
        loop_counter=0

    # if(1):
    #     if(math.sin(0.05*counter1) > 0):
    #         gym.set_dof_target_position(env0, handle_list[2], 0.785398/4)
    #         gym.set_dof_target_position(env0, handle_list[3], -0.785398/4)
            
    #         for h in handle_list[0:2]:
    #             gym.set_dof_target_position(env0, h, 2*0.785398)
    #         for h in handle_list[4:6]:
    #             gym.set_dof_target_position(env0, h, -0.785398)

    #         # gym.set_dof_target_position(env0, IW1_handle0, 90.0)
    #         # gym.set_dof_target_position(env0, IW2_handle0, 90.0)
    #     else:
    #         for h in handle_list:
    #             gym.set_dof_target_position(env0, h, 0.0)

    #         # gym.set_dof_target_position(env0, IW1_handle0, -90.0)
    #         # gym.set_dof_target_position(env0, IW2_handle0, -90.0)
    #     counter1 += 1

 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
