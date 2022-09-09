''' 
-----------------------------------------------
Explanation:
Enviornment for Multiple Traveling Salesman Problem
-----------------------------------------------
'''

import os
import sys
import random
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

from envs import Env

import numpy as np
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from pprint import pprint
from adjustText import adjust_text


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
        dist = (dy ** 2 + dx ** 2) ** 0.5 + 1e-10
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

        self.x = self.location_history[-1].x
        self.y = self.location_history[-1].y

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
        self.robot_color_list = ['red', 'blue', 'purple', 'orange', 'darkcyan', 'peru', 'brown']
        
        self._from_file()

    def _from_file(self):
        file = self.config['file'] if 'file' in self.config else None
        if file:
            xys = list()
            f = open(f"data/{file}", 'r')
            nxys = f.read().split('\n') # 'n x y'
            f.close()
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

    def _num_remaining_robots_at_base(self):
        num_remaining_robots_at_base = 0
        for robot in self.robots:
            if (
                robot.x == self.base.x
                and robot.y == self.base.y
                and not robot.is_returned_to_base
            ):
                num_remaining_robots_at_base += 1
        return num_remaining_robots_at_base

    def reset(self):
        self._get_base()
        self._get_cities()
        self._get_robots()
        state = self.get_numpy_state()
        return state

    def draw(self, path=None, pred=None):
        alpha = 0.5

        plt.cla()
        plt.xlim(-0.25 * self.config['x_max'], self.config['x_max'] + 0.25 * self.config['x_max'])
        plt.ylim(-0.25 * self.config['y_max'], self.config['y_max'] + 0.25 * self.config['y_max'])

        plt.scatter(self.base.x, self.base.y, s=50, marker='^', color='green')
        
        for idx_city, city in enumerate(self.cities):
            color = 'black' if city.visited_robot is None else self.robot_color_list[city.visited_robot.id]
            plt.scatter(city.x, city.y, s=50, marker='o', color=color)
            plt.text(city.x, city.y, f"{idx_city}", fontsize=10, color='black')

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
                plt.plot([city1.x, city2.x], [city1.y, city2.y], color=color, alpha=alpha)

            if len(robot.location_history) > 0:
                city1 = robot.location_history[0]
                plt.plot([self.base.x, city1.x], [self.base.y, city1.y], color=color, alpha=alpha)
                city2 = robot.location_history[-1]
                plt.plot([city2.x, robot.x], [city2.y, robot.y], color=color, alpha=alpha)
            else:
                plt.plot([self.base.x, robot.x], [self.base.y, robot.y], color=color, alpha=alpha)

        if not pred is None:
            pred = pred[0] # (n_nodes, n_robots)
            np.set_printoptions(precision=2)
            texts = []
            for idx_city, city in enumerate(self.cities):
                text = ""
                idx_robot = self.robots.index(city.visited_robot)
                color = self.robot_color_list[idx_robot]
                for idx_robot in range(len(self.robots)):
                    text += f"{pred[idx_city][idx_robot]:.2f} \n"

                text = plt.text(
                    city.x, city.y,
                    text, 
                    fontsize=7.5,
                    color=color
                    )
                texts.append(text)
            adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

        plt.legend(bbox_to_anchor=(0.5, 0.025, 0.5, 0.5), loc=1, borderaxespad=0., fontsize=10, framealpha=0.4)
        if not path is None:
            plt.savefig(path, transparent=True)
            plt.close()

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
                assert not robot.is_returned_to_base, f"Invaild situation, there is a robot not assigned to a city!!"
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
        n_close = self.config['num_cities'] - 1
        #n_close = 26
        #n_close = 10
        state = dict()

        scale_dist = 0.25
        scale_coord = 0.5
        state['x_robot_identity'] = np.zeros([1, len(self.robots), len(self.cities) + 1, len(self.robots)])
        # (n_batch, n_robot, n_nodes, n_robots): containing information that which robot is assigned to a task (city)
        # we are going to sum through dim=1 in network so robot identidy information is going to be removed if we don't use this information.
        state['x_a'] = np.zeros([1, len(self.robots), len(self.cities) + 1, 3])
        # (n_batch, n_robot, n_nodes, d), assignments to nodes, for base node, only d_max is considered
        state['x_b'] = np.zeros([1, len(self.cities) + 1, 2])
        # (n_batch, 1, n_nodes), distance to base of each nodes, is_base for base node, 0 is constant input
        state['coord'] = np.zeros([1, len(self.cities) + 1, 2])
        # (n_batch, 1, n_nodes), x, y coordinate of each nodes
        state['edge'] = np.zeros([1, len(self.cities), len(self.cities) + 1, 5]) 
        # (n_batch, n_cities, n_nodes, 3), distance to node, distance to base, is_targeting_base
        # -> relative pos from node, distance to node, is_targeting_base
        state['avail_node_presence'] = np.zeros([1, 1, len(self.cities) + 1])
        # (n_batch, 1, n_nodes) presence availability of each node, it is 0 once it is visited for city
        state['avail_node_action'] = np.zeros([1, len(self.robots), len(self.cities) + 1])
        # (n_batch, 1, n_nodes) action availability of each node
        state['avail_robot'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['not_in_base'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['assignment_prev'] = np.zeros([1, len(self.robots), len(self.cities) + 1])
        # (n_batch, n_robots, n_nodes) previous assignments
        state['presence_prev'] = np.zeros([1, len(self.cities), len(self.cities) + 1]) 
        # (n_batch, n_cities, n_nodes), trajectories of graph
        state['visitation'] = np.ones([1, len(self.cities) + 1, len(self.robots)]) / len(self.robots)
        # (n_batch, n_nodes, n_robots), previous visitation information

        # Setup base node
        state['avail_node_presence'][0, 0, -1] = 1
        state['coord'][0, -1, :] = np.array([self.base.x, self.base.y])
        state['x_b'][0, -1, 1] = 1.0  # is_base True for base

        # Setup each node
        for idx_city, city in enumerate(self.cities):
            state['x_b'][0, idx_city, 0] = city.distance(self.base)
            state['x_b'][0, idx_city, 1] = 0. # is_base
            state['avail_node_presence'][0, 0, idx_city] = float(not city.is_visited)
            state['coord'][0, idx_city, :] = np.array([city.x, city.y])

            if city.is_visited:
                idx_robot = self.robots.index(city.visited_robot)
                state['visitation'][:, idx_city, idx_robot] = 1.0

            # (distance to node, distance to base, is_targeting_base)
            # for _idx_city, _city in enumerate(self.cities):
            #     state['edge'][0, idx_city, _idx_city, :] = np.array([
            #         city.distance(_city), 
            #         city.distance(self.base), 
            #         0
            #         ])
            # state['edge'][0, idx_city, -1, :] = np.array([
            #     city.distance(self.base), 
            #     city.distance(self.base), 
            #     1
            #     ])

            # (relative pos from node, distance to node, is_targeting_base)
            for _idx_city, _city in enumerate(self.cities):
                state['edge'][0, idx_city, _idx_city, :] = np.array([
                    city.distance(_city), 
                    city.x - _city.x,
                    city.y - _city.y,
                    float(_city.is_visited),
                    0
                    ])
            state['edge'][0, idx_city, -1, :] = np.array([
                city.distance(self.base), 
                city.x - self.base.x,
                city.y - self.base.y,
                0,
                1
                ])

        # calculate avg, std for remaining nodes
        n_present = np.sum(state['avail_node_presence'])
        avg_coord = np.mean(state['coord'], axis=1, keepdims=True)
        std_coord = np.mean((state['coord'] - avg_coord) ** 2, axis=1, keepdims=True) ** 0.5
        # avg_coord = np.sum(
        #     state['avail_node_presence'].transpose(0, 2, 1) * state['coord'],
        #     axis=1
        #     ).reshape(1, 1, 2) / n_present
        # std_coord = 1e-3 + np.sum(
        #     state['avail_node_presence'].transpose(0, 2, 1) * (state['coord'] - avg_coord) ** 2,
        #     axis=1
        #     ).reshape(1, 1, 2) ** 0.5 / n_present
        std_dist = np.sum(std_coord ** 2) ** 0.5
        self.std_dist = std_dist
        
        state['x_b'][:, :, 0] = scale_dist * state['x_b'][:, :, 0] / std_dist
        state['coord'] = scale_coord * (state['coord'] - avg_coord) / std_coord
        
        # (distance to node, distance to base, is_targeting_base)
        # state['edge'][0,:,:,0:2] = scale_dist * state['edge'][0,:,:,0:2] / std_dist
        
        # (relative pos from node, distance to node, is_targeting_base)
        state['edge'][0,:,:,0] = scale_dist * state['edge'][0,:,:,0] / std_dist
        state['edge'][0,:,:,1:3] = scale_coord * state['edge'][0,:,:,1:3] / std_coord

        is_all_assignment_none_or_base = []
        for idx_robot, robot in enumerate(self.robots):
            if len(robot.location_history) > 1:
                for i in range(len(robot.location_history) - 1):
                    city1 = robot.location_history[i]
                    city2 = robot.location_history[i + 1]
                    if not type(city2) == Base:
                        state['presence_prev'][0, city1.id, city2.id] = 1
                    else:
                        state['presence_prev'][0, city1.id, -1] = 1

            for idx_city, city in enumerate(self.cities):
                state['x_robot_identity'][0, idx_robot, idx_city, self.robots.index(robot)] = 1.0
                state['x_a'][0, idx_robot, idx_city, 0] = scale_dist * robot.distance(city) / std_dist
                state['x_a'][0, idx_robot, idx_city, 1:] = scale_coord * np.array([robot.x - city.x, robot.y - city.y]) / std_coord
                state['avail_node_action'][0, idx_robot, idx_city] = float(city.is_available())
            state['x_robot_identity'][0, idx_robot, -1, self.robots.index(robot)] = 1.0
            state['x_a'][0, idx_robot, -1, 0] = scale_dist * robot.distance(self.base) / std_dist
            state['x_a'][0, idx_robot, -1, 1:] = scale_coord * np.array([robot.x - self.base.x, robot.y - self.base.y]) / std_coord
            
            if not robot.assigned_city is None:
                if type(robot.assigned_city) == Base:
                    state['assignment_prev'][0, idx_robot, -1] = 1
                else:
                    state['assignment_prev'][0, idx_robot, robot.assigned_city.id] = 1

            is_all_assignment_none_or_base.append(
                robot.assigned_city is None or type(robot.assigned_city) == Base
                )

            state['avail_robot'][0, 0, idx_robot] = float(not (robot.is_assigned or robot.is_returned_to_base))

        dist_sorted = np.sort(
            state['avail_node_action'][0, :, :-1] * state['x_a'][0, :, :-1, len(self.robots)]
            + (1 - state['avail_node_action'][0, :, :-1]) * np.max(state['x_a'][0, :, :-1, len(self.robots)] + 10)
            , axis=1
            )
        
        n_remaining_cities = np.sum(state['avail_node_action'][0, 0, :-1])
        dist_threshold_closest = dist_sorted[:, n_close].reshape(-1, 1)
        is_in_threshold = state['x_a'][0, :, :-1, len(self.robots)] < dist_threshold_closest
        state['avail_node_action'][0, :, :-1] = is_in_threshold * state['avail_node_action'][0, :, :-1]

        is_all_assignment_none_or_base = np.array(is_all_assignment_none_or_base).all()
        if (
            n_remaining_cities == 0
            or not is_all_assignment_none_or_base
            ):
            state['avail_node_action'][0, :, -1] = 1

        return state

    def get_state_final(self, done):
        if not done:
            return None

        state_final = np.zeros([1, len(self.cities) + 1, len(self.robots)])
        for idx_city, city in enumerate(self.cities):
            assert not city.visited_robot is None, (
                f"done: {done} if game is done, every city should be visited by some robot \n"
                + f"is_returned_to_base: {[ robot.is_returned_to_base for robot in self.robots ]}"
                )
            idx_robot = self.robots.index(city.visited_robot)
            state_final[:, idx_city, idx_robot] = 1.0
        state_final[:, -1, :] = 1.0 / len(self.robots)

        return state_final
                
    def step(self, action):
        # action is defined as target city of each robot
        cm_vectors = list()
        for idx, robot in enumerate(self.robots):
            _action = action[idx]

            if (
                robot.assigned_city is None
                and not robot.is_returned_to_base
                and _action is not None
            ):
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

            # For calculating potential based reward
            # x_avg, y_avg = robot.x, robot.y
            # if robot.location_history:
            #     for _city in robot.location_history:
            #         x_avg, y_avg = x_avg + _city.x, y_avg + _city.y
            #     x_avg, y_avg = x_avg / (len(robot.location_history) + 1), y_avg / (len(robot.location_history) + 1)
            # dist = (x_avg ** 2 + y_avg ** 2) ** 0.5
            # x_avg, y_avg = x_avg / dist, y_avg / dist
            # cm_vectors.append( (x_avg, y_avg) )

        time_spent, done = self.update_robots()
        state_next = self.get_numpy_state()
        state_final = self.get_state_final(done)        
        #state_next['state_final'] = state_final
        
        reward = - self.config['scale_reward'] * time_spent / self.std_dist

        return state_next, reward, done

    def sample_action(self):
        available_robots = [i for i, robot in enumerate(self.robots) if not robot.is_assigned and not robot.is_returned_to_base]
        available_nodes = [i for i, city in enumerate(self.cities) if not city.is_visited and city.assigned_robot is None]
        if available_nodes == []:
            available_nodes.append(len(self.cities))
        action = [None for _ in range(len(self.robots))]
        robot = random.sample(available_robots, 1)[0]
        node = random.sample(available_nodes, 1)[0]
        action[robot] = node
        return action








