
import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['MKL_SERIAL'] = 'YES'

from envs.mtsp_simple import MTSP, MTSPSimple
from models.gnn import Model

import numpy as np
import torch
import torch.multiprocessing as mp
import copy

from threading import Thread

import logging
logger = logging.getLogger(__name__)

import time


def get_data(model_shared, config, q_data, q_count, q_eps):
    
    model = model_shared

    env = eval(f"{config['env']['name']}(config['env'])")
    s = env.reset()

    num_collection = 50

    while True:
        count = q_count.get()
        eps = q_eps.get(); q_eps.put(eps)

        if count >= num_collection:
            count = count - num_collection
            _num_collection = num_collection
            q_count.put(count)
        else:
            _num_collection = count
            count = 0

        for _ in range(_num_collection):
            if np.random.rand() < eps:
                action = model.random_action(s)
            else:
                action = model.action(s)
            s_next, reward, done = env.step(action['list'])
            sards = (s, action['numpy'], reward, done, s_next)
            q_data.put(sards)

            if done:
                s = env.reset()
            else:
                s = s_next

def get_data_sarsa(model_shared, config, q_data, q_count, q_eps):
    
    model = model_shared

    env = eval(f"{config['env']['name']}(config['env'])")
    s = env.reset()
    s_prev,  action_prev, reward_prev, done_prev = None, None, None, None

    num_collection = 50

    while True:
        count = q_count.get()
        eps = q_eps.get(); q_eps.put(eps)

        if count >= num_collection:
            count = count - num_collection
            _num_collection = num_collection
            q_count.put(count)
        else:
            _num_collection = count
            count = 0

        while _num_collection > 0:
            if done_prev:
                action = None
            else:
                if np.random.rand() < eps:
                    action = model.random_action(s)
                else:
                    _time_test = time.time()
                    action = model.action(s)
                    time_test = time.time() - _time_test
                    #print(f"time_test: {time_test}")
                s_next, reward, done = env.step(action['list'])

            if not (s_prev is None or action_prev is None or reward_prev is None or done_prev is None):
                sardsa = (s_prev, action_prev['numpy'], reward_prev, done_prev, s, action)
                q_data.put(sardsa)
                _num_collection -= 1

            if done_prev:
                s = env.reset()
                s_prev,  action_prev, reward_prev, done_prev = None, None, None, None
            else:
                s_prev = s
                s = s_next
                action_prev = action
                reward_prev = reward
                done_prev = done

class Simulator:
    def __init__(self, config, model):
        self.config = config
        self.model = model # Should be sheard by model.shared_memory()
        if self.model.device == 'cpu':
            self.model_cpu = self.model
        else:
            #self.model_cpu = copy.deepcopy(self.model).cpu()
            #self.model_cpu.device = 'cpu'
            self.model_cpu = self.model
        self.model_cpu.share_memory()

        self.q_data = mp.Queue()
        self.q_count = mp.Queue()
        self.q_eps = mp.Queue()

        self.procs = []
        for _ in range(self.config['learning']['num_processes']):
            target_func = None
            learning_algorithm = self.config['learning']['algorithm']
            if learning_algorithm == 'optimal_q_learning':
                target_func = get_data 
            elif learning_algorithm == 'sarsa':
                target_func = get_data_sarsa

            assert not target_func is None, f"Invalid algorithm is set for learning: {learning_algorithm}"

            proc = mp.Process(
                target = target_func, 
                args = (self.model_cpu, self.config, self.q_data, self.q_count, self.q_eps,)
                )
            proc.start()
            self.procs.append(proc)

    def get_eps(self):
        eps_end = self.config['learning']['eps']['end']
        eps_add = self.config['learning']['eps']['add']
        half_life = self.config['learning']['eps']['half_life']
        eps = eps_end +  eps_add * half_life / (half_life + self.model.step_train)
        return eps

    def sync_model(self):
        if not self.model.device == 'cpu':
            state_dict = self.model.state_dict()
            state_dict = { key: state_dict[key].cpu() for key in state_dict }
            self.model_cpu.load_state_dict(state_dict)

    def save_to_replay_buffer(self, size):
        num_data = size
        eps = self.get_eps()
        self.sync_model()
        self.q_count.put(size)
        self.q_eps.put(eps)

        while num_data > 0:
            if num_data % 100 == 0:
                logger.debug(f"collecting data.. {num_data} are left")
                #print(f"collecting data.. {num_data} are left")
            if self.q_data.qsize() > 0 :
                sards = self.q_data.get()
                self.model.add_to_replay_buffer(sards)
                num_data = num_data - 1
            else:
                time.sleep(1e-3)

    def terminate(self):
        for proc in self.procs:
            proc.kill()




