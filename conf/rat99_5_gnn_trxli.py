import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 5,
        'num_cities': 98,
        'file': 'rat99.txt',
        'scale_distance': 1e-3,
        'scale_reward': 5e-2,
    },
    'env_test':{
        'name':'MTSP',
        'num_robots': 5,
        'num_cities': 98,
        'file': 'rat99.txt',
        'scale_distance': 1e-3,
        'scale_reward': 5e-2,
    },
    'learning':{
        'step': 1000000,
        'model': 'gnn_trxli',
        'activation': 'relu',
        'algorithm': 'optimal_q_learning',
        'sampling_method': 'prioritized',
        #'algorithm': 'sarsa',

        'base_hidden_size': 128,
        'n_head': 4,
        'n_layer': 4,

        'lr_start': 1e-4,
        'lr_end': 5e-5,
        'lr_step': 500,
        'lr_decay': 0.99,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.9,
            'end': 0.1,
            'half_life': 10000,
            },
        'gamma': 0.98,
        'size_batch': 64,
        'size_replay_buffer': 80000,
        'num_rollout': 1,
        'num_processes': 4,
    }
}