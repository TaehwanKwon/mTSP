'''
-----------------------------------------------
Explanation:
Enviornment for Multiple Traveling Salesman Problem
-----------------------------------------------
'''

import os
import sys
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from envs import Env
from envs.utils import get_sample, get_decay

import numpy as np
import matplotlib.pyplot as plt

import logging

logger = logging.getLogger(__name__)

from pprint import pprint
from adjustText import adjust_text


MAX_REWARD = 10


class Robot:
    def __init__(self, robot_id, x, y, speed):
        self.id = robot_id
        self.x = x
        self.y = y
        self.speed = speed
        self.original_speed = speed # Random traveling time is controlled by changing the speed of a robot
        self.is_assigned = False
        self.location_history = []
        self.assigned_task = None
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
        dy = self.assigned_task.y - self.y
        dx = self.assigned_task.x - self.x
        dist = (dy ** 2 + dx ** 2) ** 0.5 + 1e-10
        self.x = self.x + (dx / dist) * dt * self.speed
        self.y = self.y + (dy / dist) * dt * self.speed
        logger.debug(f"self.x: {self.x}, self.y: {self.y}")

    def assign_task(self, task, random_factor):
        self.is_assigned = True
        self.assigned_task = task
        self.remaining_distance = self.distance(task)
        if type(task) == Base:
            task.assigned_robots.append(self)
        else:
            task.assigned_robot = self
            self.speed = self.original_speed / random_factor
        self.cost += self.remaining_distance / self.speed

    def finish(self):
        self.location_history.append(self.assigned_task)
        is_at_base = type(self.assigned_task) == Base
        if not is_at_base:  # if the last assignment was not the base, finish the task.
            self.location_history[-1].finish()
            logger.debug(f"robot {self.id} finish assignment task {self.assigned_task.id}")
        else:
            logger.debug(f"robot {self.id} is returned to base")
            self.is_returned_to_base = True

        self.is_assigned = False
        self.assigned_task = None
        self.remaining_distance = -1
        self.speed = self.original_speed

        self.x = self.location_history[-1].x
        self.y = self.location_history[-1].y

        return is_at_base


class Base: # Depot
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.assigned_robots = []


class Task:
    def __init__(self, task_id, x, y, age=0):
        self.id = task_id
        self.x = x
        self.y = y
        self.is_visited = False
        self.is_returned_reward = False
        self.assigned_robot = None
        self.visited_robot = None
        self.age = age

    def distance(self, entity):
        distance = ((self.x - entity.x) ** 2
                    + (self.y - entity.y) ** 2) ** 0.5
        return distance

    def is_available(self):
        return (not self.is_visited and self.assigned_robot is None)

    def update_age(self, dt):
        self.age += dt

    def get_reward(self, type, lam=0.1):
        if self.is_visited and not self.is_returned_reward:
            self.is_returned_reward = True  # Reward should be returned only once for a task
            reward = MAX_REWARD * get_decay(type, self.age, lam)
        else:
            reward = 0
        return reward

    def finish(self):
        self.is_visited = True
        self.visited_robot = self.assigned_robot
        self.assigned_robot = None


class MRRC(Env):
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
            nxys = f.read().split('\n')  # 'n x y'
            f.close()
            for nxy in nxys:
                n, x, y = nxy.split(' ')
                xys.append((float(x), float(y)))
        else:
            xys = None

        self.xys_from_file = xys

    def _get_base(self):
        if self.xys_from_file:
            base = self.xys_from_file[0]
            self.base = Base(base[0], base[1])
        else:
            self.base = Base(self.config['base']['x'], self.config['base']['y'])

    def _get_tasks(self):
        if self.xys_from_file:
            tasks = self.xys_from_file[1:]
            self.num_tasks = len(tasks)
            self.tasks = [Task(idx, tasks[idx][0], tasks[idx][1]) for idx in range(self.num_tasks)]
            xs, ys = zip(*tasks)
            x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
            self.max_distance = ((x_max - x_min) ** 2 + (y_max - y_min) ** 2) ** 0.5
            self.config['x_max'] = x_max
            self.config['y_max'] = y_max
        else:
            self.tasks = []
            self.num_tasks = self.config['num_tasks']
            for idx in range(self.config['num_tasks']):
                x = self.config['x_max'] * np.random.rand()
                y = self.config['y_max'] * np.random.rand()
                self.tasks.append(Task(idx, x, y))
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
                    self.config['base']['x'],
                    self.config['base']['y'],
                    self.config['robot_speed']
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
        self._get_tasks()
        self._get_robots()
        state = self.get_numpy_state()
        return state

    def draw(self, path=None, pred=None):
        alpha = 0.5

        plt.cla()
        plt.xlim(-0.25 * self.config['x_max'], self.config['x_max'] + 0.25 * self.config['x_max'])
        plt.ylim(-0.25 * self.config['y_max'], self.config['y_max'] + 0.25 * self.config['y_max'])

        plt.scatter(self.base.x, self.base.y, s=50, marker='^', color='green')

        for idx_task, task in enumerate(self.tasks):
            color = 'black' if task.visited_robot is None else self.robot_color_list[task.visited_robot.id]
            plt.scatter(task.x, task.y, s=50, marker='o', color=color)
            plt.text(task.x, task.y, f" {idx_task} ({task.age:.1f})", fontsize=7, color='black')

        for idx_robot, robot in enumerate(self.robots):
            color = self.robot_color_list[idx_robot]
            plt.scatter(robot.x, robot.y, c=color, s=100, marker='x', label=f'r{idx_robot} ({robot.cost:.1f})')
            if not robot.assigned_task is None:
                task = robot.assigned_task
                plt.arrow(
                    robot.x, robot.y, task.x - robot.x, task.y - robot.y,
                    color=color, head_width=0.1, head_length=0.1, length_includes_head=True
                )

            for i in range(len(robot.location_history) - 1):
                task1 = robot.location_history[i]
                task2 = robot.location_history[i + 1]
                plt.plot([task1.x, task2.x], [task1.y, task2.y], color=color, alpha=alpha)

            if len(robot.location_history) > 0:
                task1 = robot.location_history[0]
                plt.plot([self.base.x, task1.x], [self.base.y, task1.y], color=color, alpha=alpha)
                task2 = robot.location_history[-1]
                plt.plot([task2.x, robot.x], [task2.y, robot.y], color=color, alpha=alpha)
            else:
                plt.plot([self.base.x, robot.x], [self.base.y, robot.y], color=color, alpha=alpha)

        if not pred is None:
            pred = pred[0]  # (n_nodes, n_robots)
            np.set_printoptions(precision=2)
            texts = []
            for idx_task, task in enumerate(self.tasks):
                text = ""
                idx_robot = self.robots.index(task.visited_robot)
                color = self.robot_color_list[idx_robot]
                for idx_robot in range(len(self.robots)):
                    text += f"{pred[idx_task][idx_robot]:.2f} \n"

                text = plt.text(
                    task.x, task.y,
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
        _is_returned_to_bases = []  # is_at_bases of the robots finished assignment at this transition
        for idx, robot in enumerate(self.robots):
            if robot.is_assigned:
                _distance = robot.remaining_distance
                _dt = _distance / robot.speed
            else:
                assert not robot.is_returned_to_base, f"Invaild situation, there is a robot not assigned to a task!!"
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

        for task in self.tasks:
            if not (task.is_visited and task.is_returned_reward):
                # Age of a task should be updated if it is not collected or didn't return reward
                task.update_age(dt)

        is_returned_to_bases = [robot.is_returned_to_base for robot in self.robots]
        done = np.array(is_returned_to_bases).all()
        if (
                np.array(
                    _is_returned_to_bases).all()  # At this transition, if all robots finished their assignment are in base
                and not done  # If not all robots are returned to base (some of them are doing assignment)
        ):
            dt_sum, done = self.update_robots(dt_sum, done)

        done = bool(done)
        return dt_sum, done

    def get_numpy_state(self):
        n_close = self.config['num_tasks'] - 1
        # n_close = 26
        # n_close = 10
        state = dict()

        scale_dist = 0.1
        scale_coord = 0.1
        scale_age = 0.1
        # we are going to sum through dim=1 in network so robot identidy information is going to be removed if we don't use this information.
        state['x_a'] = np.zeros([1, len(self.robots), len(self.tasks) + 1, 4])
        # (n_batch, n_robot, n_nodes, d), d (4): assignment distance from a robot (1), coordinate of a task (2), age of a task (1)
        state['x_b'] = np.zeros([1, len(self.tasks) + 1, 2])
        # (n_batch, 1, n_nodes), distance to base of each nodes, is_base for base node, 0 is constant input
        state['coord'] = np.zeros([1, len(self.tasks) + 1, 2])
        # (n_batch, 1, n_nodes), x, y coordinate of each nodes
        state['edge'] = np.zeros([1, len(self.tasks), len(self.tasks) + 1, 5])
        # (n_batch, n_tasks, n_nodes, 3), distance to node, distance to base, is_targeting_base
        # -> relative pos from node, distance to node, is_targeting_base
        state['avail_node_presence'] = np.zeros([1, 1, len(self.tasks) + 1])
        # (n_batch, 1, n_nodes) presence availability of each node, it is 0 once it is visited for task
        state['avail_node_action'] = np.zeros([1, len(self.robots), len(self.tasks) + 1])
        # (n_batch, 1, n_nodes) action availability of each node
        state['avail_robot'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['not_in_base'] = np.zeros([1, 1, len(self.robots)])
        # (n_batch, 1, n_robots) availability of robot
        state['assignment_prev'] = np.zeros([1, len(self.robots), len(self.tasks) + 1])
        # (n_batch, n_robots, n_nodes) previous assignments
        state['presence_prev'] = np.zeros([1, len(self.tasks), len(self.tasks) + 1])
        # (n_batch, n_tasks, n_nodes), trajectories of graph
        state['visitation'] = np.ones([1, len(self.tasks) + 1, len(self.robots)]) / len(self.robots)
        # (n_batch, n_nodes, n_robots), previous visitation information

        # Setup base node
        state['avail_node_presence'][0, 0, -1] = 1
        state['coord'][0, -1, :] = np.array([self.base.x, self.base.y])
        state['x_b'][0, -1, 1] = 1.0  # is_base True for base

        # Setup each node
        for idx_task, task in enumerate(self.tasks):
            state['x_b'][0, idx_task, 0] = task.distance(self.base)
            state['x_b'][0, idx_task, 1] = 0.  # is_base
            state['avail_node_presence'][0, 0, idx_task] = float(not task.is_visited)
            state['coord'][0, idx_task, :] = np.array([task.x, task.y])

            if task.is_visited:
                idx_robot = self.robots.index(task.visited_robot)
                state['visitation'][:, idx_task, idx_robot] = 1.0

            # (distance to node, distance to base, is_targeting_base)
            # for _idx_task, _task in enumerate(self.tasks):
            #     state['edge'][0, idx_task, _idx_task, :] = np.array([
            #         task.distance(_task),
            #         task.distance(self.base),
            #         0
            #         ])
            # state['edge'][0, idx_task, -1, :] = np.array([
            #     task.distance(self.base),
            #     task.distance(self.base),
            #     1
            #     ])

            # (relative pos from node, distance to node, is_targeting_base)
            for _idx_task, _task in enumerate(self.tasks):
                state['edge'][0, idx_task, _idx_task, :] = np.array([
                    scale_dist * task.distance(_task),
                    scale_coord * (task.x - _task.x),
                    scale_coord * (task.y - _task.y),
                    float(_task.is_visited),  # is_visited
                    0  # is_base
                ])
            state['edge'][0, idx_task, -1, :] = np.array([
                scale_dist * task.distance(self.base),
                scale_coord * (task.x - self.base.x),
                scale_coord * (task.y - self.base.y),
                0,  # is_visited
                1  # is_base
            ])

        is_all_assignment_none_or_base = []
        for idx_robot, robot in enumerate(self.robots):
            if len(robot.location_history) > 1:
                for i in range(len(robot.location_history) - 1):
                    task1 = robot.location_history[i]
                    task2 = robot.location_history[i + 1]
                    if not type(task2) == Base:
                        state['presence_prev'][0, task1.id, task2.id] = 1
                    else:
                        state['presence_prev'][0, task1.id, -1] = 1

            for idx_task, task in enumerate(self.tasks):
                state['x_a'][0, idx_robot, idx_task, 0] = scale_dist * robot.distance(task)
                state['x_a'][0, idx_robot, idx_task, 1:] = scale_coord * np.array(
                    [robot.x - task.x, robot.y - task.y])
                state['x_a'][0, idx_robot, idx_task, 3] = scale_age * task.age
                state['avail_node_action'][0, idx_robot, idx_task] = float(task.is_available())
            state['x_a'][0, idx_robot, -1, 0] = scale_dist * robot.distance(self.base)
            state['x_a'][0, idx_robot, -1, 1:] = scale_coord * np.array(
                [robot.x - self.base.x, robot.y - self.base.y])

            if not robot.assigned_task is None:
                if type(robot.assigned_task) == Base:
                    state['assignment_prev'][0, idx_robot, -1] = 1
                else:
                    state['assignment_prev'][0, idx_robot, robot.assigned_task.id] = 1

            is_all_assignment_none_or_base.append(
                robot.assigned_task is None or type(robot.assigned_task) == Base
            )

            state['avail_robot'][0, 0, idx_robot] = float(not (robot.is_assigned or robot.is_returned_to_base))

        dist_sorted = np.sort(
            state['avail_node_action'][0, :, :-1] * state['x_a'][0, :, :-1, len(self.robots)]
            + (1 - state['avail_node_action'][0, :, :-1]) * np.max(state['x_a'][0, :, :-1, len(self.robots)] + 10)
            , axis=1
        )

        n_remaining_tasks = np.sum(state['avail_node_action'][0, 0, :-1])
        dist_threshold_closest = dist_sorted[:, n_close].reshape(-1, 1)
        is_in_threshold = state['x_a'][0, :, :-1, len(self.robots)] < dist_threshold_closest
        state['avail_node_action'][0, :, :-1] = is_in_threshold * state['avail_node_action'][0, :, :-1]

        is_all_assignment_none_or_base = np.array(is_all_assignment_none_or_base).all()
        if (
                n_remaining_tasks == 0
                or not is_all_assignment_none_or_base
        ):
            state['avail_node_action'][0, :, -1] = 1

        return state

    def get_state_final(self, done):
        if not done:
            return None

        state_final = np.zeros([1, len(self.tasks) + 1, len(self.robots)])
        for idx_task, task in enumerate(self.tasks):
            assert not task.visited_robot is None, (
                    f"done: {done} if game is done, every task should be visited by some robot \n"
                    + f"is_returned_to_base: {[robot.is_returned_to_base for robot in self.robots]}"
            )
            idx_robot = self.robots.index(task.visited_robot)
            state_final[:, idx_task, idx_robot] = 1.0
        state_final[:, -1, :] = 1.0 / len(self.robots)

        return state_final

    def step(self, action):
        # action is defined as target task of each robot
        cm_vectors = list()
        for idx, robot in enumerate(self.robots):
            _action = action[idx]

            if (
                    robot.assigned_task is None
                    and not robot.is_returned_to_base
                    and _action is not None
            ):
                # Handle action 'go_base'
                random_factor = get_sample(self.config['random_travel'])
                if _action == len(self.tasks):
                    logger.debug(f"robot {robot.id} is going to base")
                    robot.assign_task(task=self.base, random_factor=random_factor)
                    continue

                # Assign a task to a robot
                target_task = self.tasks[_action]
                assert target_task.is_available(), (
                        f"Invalid task {target_task} is chosen,"
                        + f"is_visited: {target_task.is_visited}, assigned_robot: {target_task.assigned_robot}"
                )
                robot.assign_task(task=target_task, random_factor=random_factor)
                logger.debug(f"robot {robot.id} is assigned to task {target_task.id}")

            else:  # If robot is already assigned to a task, its action should be None
                # logger.debug(_action, robot.assigned_task, robot.is_returned_to_base)
                assert _action is None, (
                    f"Robot {idx} is not available!!"
                )

            # For calculating potential based reward
            # x_avg, y_avg = robot.x, robot.y
            # if robot.location_history:
            #     for _task in robot.location_history:
            #         x_avg, y_avg = x_avg + _task.x, y_avg + _task.y
            #     x_avg, y_avg = x_avg / (len(robot.location_history) + 1), y_avg / (len(robot.location_history) + 1)
            # dist = (x_avg ** 2 + y_avg ** 2) ** 0.5
            # x_avg, y_avg = x_avg / dist, y_avg / dist
            # cm_vectors.append( (x_avg, y_avg) )

        time_spent, done = self.update_robots()
        state_next = self.get_numpy_state()
        state_final = self.get_state_final(done)
        # state_next['state_final'] = state_final
        reward = (self.config['scale_reward'] *
                  np.sum([task.get_reward(type=self.config['reward_type']) for task in self.tasks]))

        return state_next, reward, done

    def sample_action(self):
        available_robots = [i for i, robot in enumerate(self.robots) if
                            not robot.is_assigned and not robot.is_returned_to_base]
        available_nodes = [i for i, task in enumerate(self.tasks) if
                           not task.is_visited and task.assigned_robot is None]
        if available_nodes == []:
            available_nodes.append(len(self.tasks))
        action = [None for _ in range(len(self.robots))]
        robot = random.sample(available_robots, 1)[0]
        node = random.sample(available_nodes, 1)[0]
        action[robot] = node
        return action








