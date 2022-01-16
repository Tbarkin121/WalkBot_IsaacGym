import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from tasks.base.vec_task import VecTask


class WalkBot_4DOF(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

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

        # Observations
        # # Body State : Position(3), Orientation(4), Linear Vel(3), Angular Vel(3)
        # # Joint Positions (6)
        # # Joint Velocities (6)
        # # Total = 25 obserations
        # # Will need to add drive signal observations later
        # self.cfg["env"]["numObservations"] = 25

        # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(6), num_dofs(6), num_dofs(6)
        self.cfg["env"]["numObservations"] = 36
        # Joint Target Positions (6) 
        self.cfg["env"]["numActions"] = 8

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        # get gym GPU DOF state tensor
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # get gym GPU rigid body state tensor 
        rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state = gymtorch.wrap_tensor(rb_state_tensor)
        self.rb_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., 0:15, 0:3]  #(num_envs, num_rigid_bodies, 13)[pos,ori,Lin-vel,Ang-vel]
        self.body_pos = self.rb_state.view(self.num_envs, self.num_bodies, 13)[..., self.body_dict['head'], 0:3] #num_envs, num_rigid_bodies, 13 (pos,ori,Lvel,Avel)

        # set init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = torch.tensor(state, device=self.device)
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.root_pos = self.root_states.view(self.num_envs, 1, 13)[..., 0, 0:3] #(num_envs, num_actors, 13)[pos,ori,Lin-vel,Ang-vel]
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

        # Measurements for rewards
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([0, 1000, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([0, 1, 0], device=self.device).repeat((self.num_envs, 1))

        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        self.frame_count = 0

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
        asset_file = "urdf/WalkBot_4DOF/urdf/WalkBot_4DOF.urdf"
        # asset_file = "urdf/cassie/urdf/cassie.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 10000

        walkbot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(walkbot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(walkbot_asset)
        # self.num_actor = get_sim_actor_count
        # self.num_rb = get_actor_rigid_body_count(walkbot_asset)
        
        pose = gymapi.Transform()
        # pose.p.z = 0.5
        pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        # asset is rotated z-up by default, no additional rotations needed
        pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.start_rotation = torch.tensor([pose.r.x, pose.r.y, pose.r.z, pose.r.w], device=self.device)

        self.walkbot_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            walkbot_handle = self.gym.create_actor(env_ptr, walkbot_asset, pose, "walkbot", i, 1, 0)
            
            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, walkbot_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.27, 0.1, 0.66))

            dof_props = self.gym.get_actor_dof_properties(env_ptr, walkbot_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
            dof_props['stiffness'][:] = 100000.0
            dof_props['damping'][:] = 10000.0
            dof_props['velocity'][:] = 38.0
            dof_props['effort'][:] = 0.22
            dof_props['friction'][:] = 0.0

            self.gym.set_actor_dof_properties(env_ptr, walkbot_handle, dof_props)

            self.envs.append(env_ptr)
            self.walkbot_handles.append(walkbot_handle)

        for j in range(self.num_dof):
            if dof_props['lower'][j] > dof_props['upper'][j]:
                self.dof_limits_lower.append(dof_props['upper'][j])
                self.dof_limits_upper.append(dof_props['lower'][j])
            else:
                self.dof_limits_lower.append(dof_props['lower'][j])
                self.dof_limits_upper.append(dof_props['upper'][j])
        
        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.body_dict = self.gym.get_actor_rigid_body_dict(env_ptr, walkbot_handle)
        print('RB Name List:')
        for b in self.body_dict:
                print(b)
                
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
        self.rew_buf[:], self.reset_buf[:] = compute_walkbot_reward(
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
            self.max_episode_length)


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print(self.root_states)
        # print(self.root_states.shape)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_walkbot_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions, self.dt,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        # self.obs_buf[env_ids, 0:13] = self.root_states[env_ids, :] # Position(3), Orientation(4), Linear Vel(3), Angular Vel(3)
        # self.obs_buf[env_ids, 13] = self.dof_pos[env_ids, 0].squeeze() #Joint 1 Pos
        # self.obs_buf[env_ids, 14] = self.dof_pos[env_ids, 1].squeeze() #Joint 2 Pos
        # self.obs_buf[env_ids, 15] = self.dof_pos[env_ids, 2].squeeze() # ...
        # self.obs_buf[env_ids, 16] = self.dof_pos[env_ids, 3].squeeze() # ...
        # self.obs_buf[env_ids, 17] = self.dof_pos[env_ids, 4].squeeze() # ...
        # self.obs_buf[env_ids, 18] = self.dof_pos[env_ids, 5].squeeze() # ...
        # self.obs_buf[env_ids, 19] = self.dof_vel[env_ids, 0].squeeze() # Joint 1 Vel
        # self.obs_buf[env_ids, 20] = self.dof_vel[env_ids, 1].squeeze() # Joint 2 Vel
        # self.obs_buf[env_ids, 21] = self.dof_vel[env_ids, 2].squeeze() # ...
        # self.obs_buf[env_ids, 22] = self.dof_vel[env_ids, 3].squeeze() # ...
        # self.obs_buf[env_ids, 23] = self.dof_vel[env_ids, 4].squeeze() # ...
        # self.obs_buf[env_ids, 24] = self.dof_vel[env_ids, 5].squeeze() # ...
        

        return self.obs_buf

    def reset_idx(self, env_ids):
        positions = 0.0 * (torch.rand((len(env_ids), self.num_dof), device=self.device))
        velocities = 0.0 * (torch.rand((len(env_ids), self.num_dof), device=self.device))

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

    def pre_physics_step(self, actions):
        # print('actions')
        # print(actions)
        # print(actions.shape)
        # print(actions.to(self.device).squeeze().shape())
        self.actions = actions.clone().to(self.device)
        targets = self.actions*1.5708
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

        # Apply random forces
        if (self.frame_count - 99) % 200 == 0:
            # set forces and torques for the ant root bodies
            forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
            forces[:, 0, 0:3] = torch.distributions.Uniform(-2, 2).sample(torch.Size([3]))
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(forces), gymtorch.unwrap_tensor(torques), gymapi.ENV_SPACE)
        self.frame_count += 1

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
# def compute_walkbot_reward(root_pos, reset_buf, progress_buf, max_episode_length):
#     # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

#     # reward = body_height - torch.linalg.norm(root_pos, dim=1)/2.0
#     reward = root_pos[:, 2]
#     # reward = 1 - (pole1_angle * pole1_angle)/2 - (pole2_angle * pole2_angle)/2 - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole1_vel) - 0.005 * torch.abs(pole2_vel)

#     # adjust reward for reset agents
#     # reward = torch.where(torch.abs(cart_pos) > reset_dist, torch.ones_like(reward) * -2.0, reward)
#     # reward = torch.where(torch.abs(pole1_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

#     # reset = torch.where(torch.linalg.norm(body_height, dim=1) < 0.2, torch.ones_like(reset_buf), reset_buf)
#     # reset = torch.where(torch.abs(pole1_angle) > np.pi / 2, torch.ones_like(reset_buf), reset)
#     # reset = torch.where(progress_buf >= max_episode_length, torch.ones_like(reset_buf), reset_buf)
#     reset = torch.where(root_pos[:, 2] <= 0.14, torch.ones_like(reset_buf), reset_buf)
#     reward -= reset*10 # Lose points for falling
#     # reset = reset | torch.any(body_height)
#     return reward, reset

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
    max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float) -> Tuple[Tensor, Tensor]

    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials

    # total_reward = progress_reward + alive_reward + up_reward + heading_reward
    total_reward = progress_reward

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset

@torch.jit.script
def compute_walkbot_observations(obs_buf, root_states, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             actions, dt,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), num_dofs(8)
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale,
                     actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec