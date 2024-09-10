import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from planner import Planner
from skill import GoTo_Goal, Explore, Pickup, Drop, Toggle, Wait
from mediator import IDX_TO_SKILL, IDX_TO_OBJECT

# single step (can handle soft planner)
class TeacherPolicy():
    def __init__(self, task, offline, soft, prefix, action_space, agent_view_size):
        self.planner = Planner(task, offline, soft, prefix)
        self.agent_view_size = agent_view_size
        self.action_space = action_space
        
    def get_skill_name(self, skill):
        try:
            return IDX_TO_SKILL[skill["action"]] + " " + IDX_TO_OBJECT[skill["object"]]
        except AttributeError:
            return "None"
        
    def reset(self):
        self.skill = None
        self.skill_list = []
        self.skill_teminated = False
        self.planner.reset() 

    def skill2teacher(self, skill):
        skill_action = skill['action']
        if skill_action == 0:
            teacher = Explore(self.agent_view_size)
        elif skill_action == 1:
            teacher = GoTo_Goal(skill['coordinate'])
        elif skill_action == 2:
            teacher = Pickup(skill['object'])
        elif skill_action == 3:
            teacher = Drop(skill['object'])
        elif skill_action == 4:
            teacher = Toggle(skill['object'])
        elif skill_action == 6:
            teacher = Wait()
        else:
            assert False, "invalid skill"
        return teacher
    
    def get_action(self, skill_list, obs):
        teminated = True
        action = None
        while not action and teminated and len(skill_list) > 0:
            skill = skill_list.pop(0)
            teacher = self.skill2teacher(skill)
            action, teminated = teacher(obs)
                
        if action == None:

            action = 6
            
        action = np.array([i == action for i in range(self.action_space)], dtype=np.float32)
            
        return action
    
    def __call__(self, obs):
        skill_list, probs = self.planner(obs)
        action = np.zeros(self.action_space)
        for skills, prob in zip(skill_list, probs):
            action += self.get_action(skills, obs) * prob
        return action

