import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 98,
        'file': 'rat99.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 98,
        'file': 'rat99.txt',
        'scale_distance': 5e-4,
        'scale_reward':2.5e-4,
    },
    'learning':{
        'step': 100000,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
        'lr': 1e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 5000,
            },
        'gamma': 1.0,
        'size_batch': 16,
        'size_replay_buffer': 1000,
        'num_rollout':5,
        'num_processes': 1,
    }
}