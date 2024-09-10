from abc import ABC, abstractmethod
import os
import torch
from .model import MLPBase, LSTMBase

class Base(ABC):
    """The base class for RL algorithms."""

    def __init__(self, model, obs_space, action_space, device, save_path, recurrent):
        self.device = device
        self.save_path = save_path
        self.recurrent = recurrent
        
        if model:
            self.model = model.to(self.device)
        elif self.recurrent:
            print("use LSTM......")
            self.model = LSTMBase(obs_space, action_space).to(self.device)
        else:
            print("use MLP......")
            self.model = MLPBase(obs_space, action_space).to(self.device)
        
    def save(self, name="acmodel"):
        try:
            os.makedirs(self.save_path)
        except OSError:
            pass
        filetype = ".pt" # pytorch model
        torch.save(self.model, os.path.join(self.save_path, name + filetype))

    @abstractmethod
    def update_policy(self):
        pass