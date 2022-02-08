import gym
from gym import spaces
import numpy as np
from tasks.base.jet_task import JetTask
import torch
# import yaml
# from rl_games.torch_runner import Runner
# import os
# from collections import deque

class WalkBot_Hard(JetTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]
        self.cfg["env"]["numObservations"] = 15
        self.cfg["env"]["numActions"] = 3
        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)

        
    def pre_physics_step(self, actions):
        print(actions)
        self.obs_buf = torch.rand((self.num_envs, self.num_observations), device=self.device)
        

    def post_physics_step(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        # print('Resetting IDX! Env_IDs = {}'.format(env_ids))
        self.reset_buf[env_ids] = 0