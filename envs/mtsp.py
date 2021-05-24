''' 
-----------------------------------------------
Explanation:
Enviornment for Multiple Traveling Salesman Problem
-----------------------------------------------
'''

import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

from envs import Env

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from pprint import pprint


class Robot:
    def __init__(self, robot_id, x, y, speed):
        self.id = robot_id
        self.x = x
        self.y = y
        self.speed = speed
        self.is_assigned = False
        self.location_history = []
        self.assigned_city = None
        self.remaining_distance = -1
        self.is_returned_to_base = False
        self.cost = 0

    def distance(self, entity):
        distance = (
            (self.x - entity.x) ** 2 
            + (self.y - entity.y) ** 2
            ) ** 0.5
        return distance

    def update_distance(self, dt):
        self.remaining_distance = self.remaining_distance - dt * self.speed
        dy = self.assigned_city.y - self.y  
        dx = self.assigned_city.x - self.x
        dist = (dy ** 2 + dx ** 2) ** 0.5
        self.x = self.x + (dx / dist) * dt * self.speed
        self.y = self.y + (dy / dist) * dt * self.speed
        logger.debug(f"self.x: {self.x}, self.y: {self.y}")

    def assign_city(self, city):
        self.is_assigned = True
        self.assigned_city = city
        self.remaining_distance = self.distance(city)
        if type(city) == Base:
           city.assigned_robots.append(self) 
        else:
            city.assigned_robot = self
        self.cost += self.remaining_distance

    def finish(self):
        self.location_history.append(self.assigned_city)
        is_at_base = type(self.assigned_city) == Base
        if not is_at_base: # if the last assignment was not the base, finish the city.
            self.location_history[-1].finish()
            logger.debug(f"robot {self.id} finish assignment city {self.assigned_city.id}")
        else:
            logger.debug(f"robot {self.id} is returned to base")
            self.is_returned_to_base = True
            
        self.is_assigned = False
        self.assigned_city = None
        self.remaining_distance = -1

        return is_at_base

class Base:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.assigned_robots = []

class City:
    def __init__(self, city_id, x, y):
        self.id = city_id
        self.x = x
        self.y = y
        self.is_visited = False
        self.assigned_robot = None
        self.visited_robot = None

    def distance(self, entity):
        distance = (
            (self.x - entity.x) ** 2 
            + (self.y - entity.y) ** 2
            ) ** 0.5
        return distance

    def is_available(self):
        return (not self.is_visited and self.assigned_robot is None)

    def finish(self):
        self.is_visited = True
        self.visited_robot = self.assigned_robot
        self.assigned_robot = None


class MTSP(Env):
    def __init__(self, config_env):
        self.config = config_env
        self.finished_robots = []
        self.robot_color_list = ['red', 'blue', 'purple', 'orange', 'sky', 'yellow', 'pink']
        
        self._from_file()

    def _from_file(self):
        file = self.config['file'] if 'file' in self.config else None
        if file:
            xys = list()
            f = open(f"data/{file}", 'r')
            nxys = f.read().split('\n') # 'n x y'
            for nxy in nxys:
                n, x, y = nxy.split(' ')
                xys.append( (float(x), float(y)) )
        else:
            xys = None

        self.xys_from_file = xys

    def _get_base(self):
        if self.xys_from_file:
            base = self.xys_from_file[0]
            self.base = Base(base[0], base[1])
        else:
            self.base = Base(self.config['base']['x'], self.config['base']['y'])

    def _get_cities(self):
        if self.xys_from_file:
            cities = self.xys_from_file[1:]
            self.num_cities = len(cities)
            self.cities = [ City(idx, cities[idx][0], cities[idx][1]) for idx in range(self.num_cities) ]
            xs, ys = zip(*cities)
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            self.max_distance = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5
            self.config['x_max'] = x_max
            self.config['y_max'] = y_max
        else:
            self.cities = []
            self.num_cities = self.config['num_cities'] 
            for idx in range(self.config['num_cities']):
                x = self.config['x_max'] * np.random.rand()
                y = self.config['y_max'] * np.random.rand()
                self.cities.append( City(idx, x, y) )
            self.max_distance = (self.config['x_max'] ** 2 + self.config['y_max'] ** 2) ** 0.5

    def _get_robots(self):
        if self.xys_from_file:
            base = self.xys_from_file[0]
            self.robots = [ 
                Robot(
                    idx,
                    base[0], 
                    base[1],
                    1,
                    ) for idx in range(self.config['num_robots'])
                ]
        else:
            self.robots = [ 
                Robot(
                    idx,
                    self.config['robot']['x'], 
                    self.config['robot']['y'],
                    self.config['robot']['speed']
                    ) for idx in range(self.config['num_robots'])
                ]

    def reset(self):
        self._get_base()
        self._get_cities()
        self._get_robots()
        state = self.get_numpy_state()
        return state

    def draw(self, path=None):
        plt.cla()
        plt.xlim(-0.25 * self.config['x_max'], self.config['x_max'] + 0.25 * self.config['x_max'])
        plt.ylim(-0.25 * self.config['y_max'], self.config['y_max'] + 0.25 * self.config['y_max'])

        plt.scatter(self.base.x, self.base.y, s=50, marker='^', color='green')
        
        for city in self.cities:
            color = 'black' if city.visited_robot is None else self.robot_color_list[city.visited_robot.id]
            plt.scatter(city.x, city.y, s=50, marker='o', color=color)

        for idx_robot, robot in enumerate(self.robots):
            color = self.robot_color_list[idx_robot]
            plt.scatter(robot.x, robot.y, c=color, s=100, marker='x', label=f'r{idx_robot} ({robot.cost:.1f})')
            if not robot.assigned_city is None:
                city = robot.assigned_city
                plt.arrow(
                    robot.x, robot.y, city.x - robot.x, city.y - robot.y, 
                    color=color, head_width=0.1, head_length=0.1, length_includes_head=True
                    )

            for i in range(len(robot.location_history) - 1):
                city1 = robot.location_history[i]
                city2 = robot.location_history[i+1]
                plt.plot([city1.x, city2.x], [city1.y, city2.y], color=color)

            if len(robot.location_history) > 0:
                city1 = robot.location_history[0]
                plt.plot([self.base.x, city1.x], [self.base.y, city1.y], color=color)
                city2 = robot.location_history[-1]
                plt.plot([city2.x, robot.x], [city2.y, robot.y], color=color)
            else:
                plt.plot([self.base.x, robot.x], [self.base.y, robot.y], color=color)

        plt.legend(bbox_to_anchor=(0.5, 0.025, 0.5, 0.5), loc=1, borderaxespad=0., fontsize=10, framealpha=0.4)
        if not path is None:
            plt.savefig(path)

    def render(self):
        self.draw()
        plt.show()

    # update position until next assignment
    def update_robots(self, dt_sum=0, done=False):
        dt_max = 99999
        dts = np.zeros([len(self.robots)])
        _is_returned_to_bases = [] # is_at_bases of the robots finished assignment at this transition
        for idx, robot in enumerate(self.robots):
            if robot.is_assigned:
                _distance = robot.remaining_distance
                _dt = _distance / robot.speed
            else:
                if not robot.is_returned_to_base:
                    logger.error(f"Invaild situation, there is a robot not assigned to a city!!")
                _dt = np.inf
            dts[idx] = _dt

        idx_earliest_robot = dts.argmin()
        earliest_robot = self.robots[idx_earliest_robot]
        dt = dts[idx_earliest_robot]
        logger.debug(f"dts: {dts}")

        earliest_robot.update_distance(dt)
        is_returned_to_base = earliest_robot.finish()
        _is_returned_to_bases.append(is_returned_to_base)

        dt_sum += dt

        for robot in self.robots:
            if robot.is_assigned:
                robot.update_distance(dt)
                # There could be many robots who reaches its target at the same time
                if robot.remaining_distance == 0:
                    is_returned_to_base = robot.finish()
                    _is_returned_to_bases.append(is_returned_to_base)

        is_returned_to_bases = [ robot.is_returned_to_base for robot in self.robots ]
        done = np.array(is_returned_to_bases).all()
        if ( 
            np.array(_is_returned_to_bases).all() # At this transition, if all robots finished their assignment are in base
            and not done # If not all robots are returned to base (some of them are doing assignment)
            ):
            dt_sum, done = self.update_robots(dt_sum, done)
            
        done = bool(done)
        return dt_sum, done

    def get_numpy_state(self):
        state = dict()

        state['x_a'] = np.zeros([1, len(self.robots), len(self.cities) + 1])
        # (n_batch, n_robot, n_nodes), assignments to nodes, for base node, only d_max is considered
        state['x_b'] = np.zeros([1, len(self.cities) + 1, 1])
        # (n_batch, 1, n_nodes), distance to base of each nodes, for base node, 0 is constant input
        state['edge'] = np.zeros([1, len(self.cities), len(self.cities) + 1, 3]) 
        # (n_batch, n_cities, n_nodes, 3), distance to node, distance to base, is_targeting_base
        state['avail_node_presence'] = np.zeros([1, 1, len(self.cities) + 1])
        # (n_batch, 1, n_nodes) presence availability of each node, it is 0 once it is visited for city
        state['avail_node_action'] = np.zeros([1, 1, len(self.cities) + 1])
        # (n_batch, 1, n_nodes) action availability of each node
        state['avail_robot'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['not_in_base'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['assignment_prev'] = np.zeros([1, len(self.robots), len(self.cities) + 1])
        # (n_batch, n_robots, n_nodes) previous assignments

        is_all_assignment_none_or_base = []
        for idx_robot, robot in enumerate(self.robots):
            for idx_city, city in enumerate(self.cities):
                state['x_a'][0, idx_robot, idx_city] = self.config['scale_distance'] * robot.distance(city)
            state['x_a'][0, idx_robot, -1] = self.config['scale_distance'] * robot.distance(self.base)
            
            if not robot.assigned_city is None:
                if type(robot.assigned_city) == Base:
                    state['assignment_prev'][0, idx_robot, -1] = 1
                else:
                    state['assignment_prev'][0, idx_robot, robot.assigned_city.id] = 1

            is_all_assignment_none_or_base.append(
                robot.assigned_city is None or type(robot.assigned_city) == Base
                )

            state['avail_robot'][0, 0, idx_robot] = float(not (robot.is_assigned or robot.is_returned_to_base))

        for idx_city, city in enumerate(self.cities):
            state['x_b'][0, idx_city, 0] = self.config['scale_distance'] * city.distance(self.base)
            state['avail_node_presence'][0, 0, idx_city] = float(not city.is_visited)
            state['avail_node_action'][0, 0, idx_city] = float(city.is_available())

            for _idx_city, _city in enumerate(self.cities):
                state['edge'][0, idx_city, _idx_city, :] = np.array([
                    self.config['scale_distance'] * city.distance(_city), 
                    self.config['scale_distance'] * city.distance(self.base), 
                    0
                    ])
            state['edge'][0, idx_city, -1, :] = np.array([
                self.config['scale_distance'] * city.distance(self.base), 
                self.config['scale_distance'] * city.distance(self.base), 
                1
                ])

        n_remained_cities = np.sum(state['avail_node_action'])
        is_all_assignment_none_or_base = np.array(is_all_assignment_none_or_base).all()
        if (
            n_remained_cities == 0
            or not is_all_assignment_none_or_base
            ):
            state['avail_node_action'][0, 0, -1] = 1
        state['avail_node_presence'][0, 0, -1] = 1

        return state
                
    def step(self, action):
        # action is defined as target city of each robot
        for idx, robot in enumerate(self.robots):
            _action = action[idx]

            if robot.assigned_city is None and not robot.is_returned_to_base:
                # Handle action 'go_base'
                if _action == len(self.cities):
                    logger.debug(f"robot {robot.id} is going to base")
                    robot.assign_city(self.base)
                    continue

                # Assign a city to a robot                
                target_city = self.cities[_action]
                assert target_city.is_available(), (
                    f"Invalid city {target_city} is chosen,"
                    + f"is_visited: {target_city.is_visited}, assigned_robot: {target_city.assigned_robot}"
                    )
                robot.assign_city(target_city)
                logger.debug(f"robot {robot.id} is assigned to city {target_city.id}")

            else: # If robot is already assigned to a city, its action should be None
                #logger.debug(_action, robot.assigned_city, robot.is_returned_to_base)
                assert _action is None, (
                    f"Robot {idx} is not available!!"
                    )
        time_spent, done = self.update_robots()
        state_next = self.get_numpy_state()
        
        reward = - self.config['scale_reward'] * time_spent

        return state_next, reward, done

        







