import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 20,
        'file': 'berlin21.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 20,
        'file': 'berlin21.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'learning':{
        'step': 100000,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
        'lr_start': 5e-4,
        'lr_end': 1e-4,
        'lr_step': 500,
        'lr_decay': 0.99, 
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 5000,
            },
        'gamma': 1.0,
        'size_batch': 128,
        'size_replay_buffer': 10000,
        'num_rollout':1,
        'num_processes': 4,
    }
}