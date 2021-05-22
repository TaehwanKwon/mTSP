
import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

from envs.mtsp_simple import MTSP, MTSPSimple

import numpy as np
import multiprocessing as mp
import copy

def get_data(model, config, q_data, q_count, q_eps):
    env = eval(f"{config['env']['name']}(config)")
    s = env.reset()

    num_collection = 20

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
            sarsd = (s, action['numpy'], reward, s_next, done)
            q_data.put(sarsd)

            if done:
                s = env.reset()
            else:
                s = s_next

class Simulator:
    def __init__(self, config, model):
        self.config = config
        self.model = model # Should be sheard by model.shared_memory()
        if self.model.device == 'cpu':
            self.model_cpu = self.model
        else:
            self.model_cpu = copy.deepcopy(self.model).cpu()
        self.model_cpu.share_memory()

        self.q_data = mp.Queue()
        self.q_count = mp.Queue()
        self.q_eps = mp.Queue()

        self.procs = []
        for _ in range(self.config['learning']['num_processes']):
            proc = mp.Process(
                target=get_data, 
                args=(self.model_cpu, self.config, self.q_data, self.q_count, self.q_eps,)
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
            if self.q_data.qsize() > 0:
                sarsd = self.q_data.get()
                self.model.add_to_replay_buffer(sarsd)
                num_data = num_data - 1

    def terminate(self):
        for proc in self.procs:
            proc.terminate()


