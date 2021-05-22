import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

config = {
    'env':{
        'name':'MSTP',
        'num_robots': 4,
        'num_cities': 50, #'file': 'eli51.txt' # Comment this line if you don't want to set this
        'x_max': 70,
        'y_max': 70,
        'robot':{
            'x': 5, 
            'y':5,
            'speed': 5,
        },
        'base':{
            'x': 5, 'y':5,
        },
        'scale_distance': 0.01,
        'scale_reward':0.05,
    },
    'learning':{
        'lr': 1e-4,
        'eps': 0.2, 
    }
}