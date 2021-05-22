''' 
-----------------------------------------------
Explanation:
Enviornment for Multiple Traveling Salesman Problem
-----------------------------------------------
'''

import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

class QlearningAgent:
    def __init__(self, config):
        self.config = config['learning']

    def get_loss(self, processed_batch):
        Q_target = processed_batch['reward'] + processed_batch['Q']
        Q_next_max = processed_batch['Q_next_max']
        loss = F.l1_smooth_loss(Q_target, Q_next)

        return loss
