import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 2,
        'num_cities': 20, #'file': 'eli51.txt' # Comment this line if you don't want to set this
        'x_max': 20,
        'y_max': 20,
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
    # 'env_test':{
    #     'name':'MTSP',
    #     'num_robots': 2,
    #     'num_cities': 20,
    #     'file': 'berlin21.txt',
    #     'scale_distance': 1e-3,
    #     'scale_reward':1e-3,
    # },
    'learning':{
        'step': 100000,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
        'lr': 2.5e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 5000,
            },
        'gamma': 1.0,
        'size_batch': 128,
        'size_replay_buffer': 128,
        'num_rollout':5,
        'num_processes': 4,
    }
}