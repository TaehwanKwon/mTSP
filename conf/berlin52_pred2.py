import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 3,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 1e-3,
        'scale_reward':5e-2,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 3,
        'num_cities': 51,
        'file': 'berlin52.txt',
        'scale_distance': 1e-3,
        'scale_reward':5e-2,
    },
    'learning':{
        'step': 800000,
        'model': 'gnn_pred2',
        'presence_prev':False,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
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
        'num_processes': 2,
    }
}
