import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.core.constants import DIR_TO_VEC

'''
OBJECT_TO_IDX = {
    "unseen": 0,
    "empty": 1,
    "wall": 2,
    "floor": 3,
    "door": 4,
    "key": 5,
    "ball": 6,
    "box": 7,
    "goal": 8,
    "lava": 9,
    "agent": 10,
}

STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
'''

DIRECTION = {
    0: [1, 0],
    1: [0, 1],
    2: [-1, 0],
    3: [0, -1],
}

class BaseSkill():
    def __init__(self):
        pass
        
    def unpack_obs(self, obs):
        self.obs = obs
        agent_map = obs[:, :, 3]
        self.agent_pos = np.argwhere(agent_map != 4)[0]
        self.agent_dir = obs[self.agent_pos[0], self.agent_pos[1], 3]
        self.map = obs[:, :, 0].copy()
        self.carrying = self.map[self.agent_pos[0], self.agent_pos[1]]
        self.map[self.agent_pos[0], self.agent_pos[1]] = 10
        
        
