import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'file': 'berlin52.txt',
        'robot':{
            'x': 0, 
            'y': 5,
            'speed': 5,
        },
        'base':{
            'x': 0, 
            'y': 5,
        },
        'scale_distance': 0.05,
        'scale_reward':0.05,
    },
    'learning':{
        'step': 100000,
        'algorithm': 'optimal_q_learning',
        'lr': 1e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 5000,
            },
        'gamma': 1.0,
        'size_batch': 128,
        'size_replay_buffer': 20000,
        'num_processes': 5,
    }
}