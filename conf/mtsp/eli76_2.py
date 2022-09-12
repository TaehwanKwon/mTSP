import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 75,
        'file': 'eli76.txt',
        'scale_distance': 1e-3,
        'scale_reward':5e-2,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 75,
        'file': 'eli76.txt',
        'scale_distance': 1e-3,
        'scale_reward':5e-2,
    },
    'learning':{
        'step': 800000,
        'model': 'gnn',
        'presence_prev':False,
        'algorithm': 'optimal_q_learning',
        'sampling_method': 'prioritized', # uniform / prioritized
        #'algorithm': 'sarsa',

        'base_hidden_size': 128,

        'lr_start': 1e-5,
        'lr_end': 1e-5,
        'lr_step': 500,
        'lr_decay': 0.99, 
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.95,
            'end': 0.05,
            'half_life': 10000,
            },
        'gamma': 1.0,
        'size_batch': 64,
        'size_replay_buffer': 40000,
        'num_rollout':1,
        'num_processes': 4,
    }
}
