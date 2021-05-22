import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSP',
        'num_robots': 1,
        'num_cities': 5, #'file': 'eli51.txt' # Comment this line if you don't want to set this
        'x_max': 70,
        'y_max': 70,
        'robot':{
            'x': 5, 
            'y':5,
            'speed': 5,
        },
        'base':{
            'x': 5, 
            'y':5,
        },
        'scale_distance': 0.01,
        'scale_reward':0.05,
    },
    'learning':{
        'lr': 1e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 10000,
            },
        'gamma': 1.0,
        'size_batch': 1,
        'size_replay_buffer': 10,
        'num_processes': 2,
    }
}