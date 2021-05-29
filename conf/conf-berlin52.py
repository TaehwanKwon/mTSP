import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'learning':{
        'step': 200000,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
        'lr': 1e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.9,
            'end': 0.1,
            'half_life': 20000,
            },
        'gamma': 1.0,
        'size_batch': 64,
        'size_replay_buffer': 5000,
        'num_rollout':5,
        'num_processes': 2,
    }
}