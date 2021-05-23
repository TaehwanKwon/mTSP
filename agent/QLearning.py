''' 
-----------------------------------------------
Explanation:
Enviornment for Multiple Traveling Salesman Problem
-----------------------------------------------
'''

import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

class Agent:
    def __init__(self, config):
        self.config = config['learning']

    def get_loss(self, processed_batch):
        Q = processed_batch['Q']
        Q_next_max = processed_batch['Q_next_max']
        Q_target = processed_batch['reward'] + self.config['gamma'] * (1 - processed_batch['done']) * Q_next_max
        
        loss = F.smooth_l1_loss(Q_target, Q)

        return loss
