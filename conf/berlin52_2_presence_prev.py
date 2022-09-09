import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 1e-3,
        'scale_reward':2.5e-4,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 1e-3,
        'scale_reward':2.5e-4,
    },
    'learning':{
        'step': 200000,
        'model': 'gnn',
        'presence_prev':True,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',

        'base_hidden_size': 128,

        'lr_start': 1e-4,
        'lr_end': 1e-4,
        'lr_step': 500,
        'lr_decay': 0.99, 
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.9,
            'end': 0.1,
            'half_life': 10000,
            },
        'gamma': 1.0,
        'size_batch': 64,
        'size_replay_buffer': 10000,
        'num_rollout':1,
        'num_processes': 2,
    }
}