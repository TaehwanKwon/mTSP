''' 
-----------------------------------------------
Explanation:
This ia a very simple mtsp environment to verify that our implementation on model & algorithm.
-----------------------------------------------
'''

import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

from envs.mtsp import MTSP, Base, City, Robot



class MTSPSimple(MTSP):
    def __init__(self, config_env):
        super().__init__(config_env)
        self.config['scale_distance'] = 0.1
        self.config['scale_reward'] = 1.
    def _get_base(self):
        self.base = Base(0, 5)

    def _get_cities(self):
        self.cities = [City(0, 0,0), City(1, 10,0), City(2, 10,10), City(3, 0,10)]
        self.config['x_max'] = 10
        self.config['y_max'] = 10
        self.config['num_cities'] = len(self.cities)

    def _get_robots(self):
        self.robots = [Robot(0, 0, 5, 5)]
        self.config['num_robots'] = len(self.robots)


# |
# City3 (0,10)         City2 (10, 10)
# |
# |
# |
# |
# Robot(0,5)
# |
# |
# |
# |
# City0 (0,0)__________City1 (10, 0)________