import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MTSPSimple',
        'num_robots': 1,
        'num_cities': 4, #'file': 'eli51.txt' # Comment this line if you don't want to set this
        'x_max': 10,
        'y_max': 10,
        'robot':{
            'x': 0, 
            'y':5,
            'speed': 5,
        },
        'base':{
            'x': 0, 
            'y':5,
        },
        'scale_distance': 0.1,
        'scale_reward':0.25,
    },
    'learning':{
        'step': 4000,
        'algorithm': 'optimal_q_learning',
        #'algorithm': 'sarsa',
        'lr': 5e-4,
        'eps': { # eps = eps_end + eps_add * half_life / (half_life + training_step)
            'add': 0.45,
            'end': 0.05,
            'half_life': 1000,
            },
        'gamma': 1.0,
        'size_batch': 128,
        'size_replay_buffer': 1000,
        'num_processes': 1,
    }
}