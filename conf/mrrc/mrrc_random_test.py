import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MRRC',
        'num_robots': 2,
        'num_tasks': 20,
        'scale_distance': 1e-3,
        'scale_reward': 5e-2,
        'reward_type': 'exponential',
        'random_travel': 'exponential',
        'base': {'x': 5, 'y': 5},
        'x_max': 10,
        'y_max': 10,
        'robot_speed': 1,
    },
    'env_test':{
        'name':'MRRC',
        'num_robots': 2,
        'num_tasks': 20,
        'scale_distance': 1e-3,
        'scale_reward':5e-2,
        'reward_type': 'exponential',
        'random_travel': 'exponential',
        'base': {'x': 5, 'y': 5},
        'x_max': 10,
        'y_max': 10,
        'robot_speed': 1,
    },
    'learning':{
        'step': 500000,
        'model': 'gnn',
        'activation': 'relu',
        'algorithm': 'optimal_q_learning',
        'sampling_method': 'prioritized',
        #'algorithm': 'sarsa',

        'base_hidden_size': 128,

        'lr_start': 5e-5,
        'lr_end': 5e-5,
        'lr_step': 500,
        'lr_decay': 0.99,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.95,
            'end': 0.05,
            'half_life': 5000,
            },
        'gamma': 1.0,
        'size_batch': 64,
        'size_replay_buffer': 20000,
        'num_rollout':1,
        'num_processes': 4,
    }
}