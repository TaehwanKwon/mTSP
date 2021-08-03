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
        info = dict()
        Q = processed_batch['Q']
        Q_next_max = processed_batch['Q_next_max']
        Q_target = processed_batch['reward'] + self.config['gamma'] * (1 - processed_batch['done']) * Q_next_max
        
        loss_bellman = F.smooth_l1_loss(Q_target, Q)
        info['loss_bellman'] = loss_bellman.detach().cpu()
        loss = loss_bellman

        if 'state_final' in processed_batch:
            pred = processed_batch['pred']
            state_final = processed_batch['state_final']
            loss_cross_entropy = -torch.mean(state_final * torch.log(pred + 1e-10))
            loss = loss + 1e-3 * loss_cross_entropy
            info['loss_cross_entropy'] = loss_cross_entropy.detach().cpu()
        else:
            info['loss_cross_entropy'] = 0.

        return loss, info
