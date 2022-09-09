
import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['MKL_SERIAL'] = 'YES'

from envs.mtsp_simple import MTSP, MTSPSimple

import numpy as np
import torch
import torch.multiprocessing as mp
import copy

from threading import Thread

import logging
logger = logging.getLogger(__name__)

import time

def get_argmax_action(model, done_tuple, state_next_tuple, action):

    argmax_action_list = []
    for idx, state_next in enumerate(state_next_tuple):
        if not done_tuple[idx]:
            argmax_action = model.action(state_next)
        else:
            argmax_action = {'numpy': np.zeros([ *action.shape ])} # add an unused dummy action if the game is done
        argmax_action_list.append(argmax_action['numpy'])
    
    return argmax_action_list

def get_data(idx, config, q_data, q_data_argmax, q_count, q_eps, q_flag_models, q_model):
    device = idx % torch.cuda.device_count()
    model = __import__(f"models.{config['learning']['model']}", fromlist=[None]).Model(config, device).to(device)

    num_collection = 100
    rollout = {
        'state': list(),
        'action': list(),
        'reward': list(),
        'done': list(),
        'sards_s':list(),
    }

    env = eval(f"{config['env']['name']}(config['env'])")
    s = env.reset()
    score = 0.
    while True:
        if q_count.qsize() > 0:
            count = q_count.get()
            eps = q_eps.get(); q_eps.put(eps)

            if count >= num_collection:
                count = count - num_collection
                _num_collection = num_collection
                q_count.put(count)
            else:
                _num_collection = int(count)
                count = 0

            flag_models = q_flag_models.get()
            if flag_models[idx]:
                flag_models[idx] = False
                q_flag_models.put(flag_models)
                state_dict_cpu = q_model.get()
                q_model.put(state_dict_cpu)
                state_dict_gpu = {key: state_dict_cpu[key].to(device) for key in state_dict_cpu}
                model.load_state_dict(state_dict_gpu)
            else:
                q_flag_models.put(flag_models)

            num_data = 0
            num_remaining_data = len(rollout['sards_s'])
            done = (num_remaining_data >= _num_collection) # initiall check whether do we have to collect or not
            _rollout = dict(state=list(), action=list(), reward=list())
            while num_data < _num_collection - num_remaining_data or not done:
                _time_test = time.time()
                if np.random.rand() < eps:
                    #action = model.action(s, softmax=True)
                    action = model.random_action(s)
                else:
                    action = model.action(s, softmax=False)
                #print(f"time_test: {time.time() - _time_test}")
                s_next, reward, done = env.step(action['list'])
                score += reward
                
                rollout['state'].append(s)
                rollout['action'].append(action['numpy'])
                rollout['reward'].append(reward)
                rollout['done'].append(done)
                
                # if not done and len(rollout['state'])==config['learning']['num_rollout']:
                #     sards = (rollout['state'][0], rollout['action'][0], sum(rollout['reward']), done, s_next)
                #     rollout['state'].pop(0)
                #     rollout['action'].pop(0)
                #     rollout['reward'].pop(0)
                #     rollout['sards'].append(sards)
                #     num_data += 1
                # elif done:
                #     for i in range(config['learning']['num_rollout']):
                #         sards = (rollout['state'][0], rollout['action'][0], sum(rollout['reward']), done, s_next)
                #         rollout['state'].pop(0)
                #         rollout['action'].pop(0)
                #         rollout['reward'].pop(0)
                #         rollout['sards'].append(sards)
                #         num_data += 1

                #     assert len(rollout['state'])==0, f"the length of left state should be zero, currently {len(rollout['state'])}"

                if done:
                    #assert not s_next['state_final'] is None, "if game is done, state final shold exist"
                    rollout['state'].append(s_next)
                    for i in range(len(rollout['state']) - 1):
                        _s, _a, _r, _done, _s_next = (
                            rollout['state'][i], 
                            rollout['action'][i], 
                            rollout['reward'][i], 
                            rollout['done'][i],
                            rollout['state'][i + 1],
                            )
                        
                        if 'state_final' in s_next:
                            _s_next['state_final'] = s_next['state_final']
                            
                        sards = (_s, _a, _r, _done, _s_next, score)
                        rollout['sards_s'].append([sards, score])
                        num_data += 1
                    
                    # for _ in range(10):
                    #     rollout['sards'].append(sards)
                    #     num_data += 1
                        
                    rollout['state'].clear()
                    rollout['action'].clear()
                    rollout['reward'].clear()
                    rollout['done'].clear()

                if done:
                    s = env.reset()
                    score = 0
                else:
                    s = s_next

            for _ in range(_num_collection):
                sards_s = rollout['sards_s'].pop(0)
                q_data.put(sards_s)
        
        elif q_data_argmax.qsize() > 0:
            idx_data, data_argmax =  q_data_argmax.get()
            _done_tuple, _state_next_tuple, _action = data_argmax
            argmax_action_list = get_argmax_action(model, _done_tuple, _state_next_tuple, _action)
            q_data.put( (idx_data, argmax_action_list) )
        time.sleep(1e-3)

def get_data2(idx, config, q_data, q_count, q_eps, q_flag_models, q_model):    
    model = model_shared

    env = eval(f"{config['env']['name']}(config['env'])")
    s = env.reset()
    s_prev,  action_prev, reward_prev, done_prev = None, None, None, None

    num_collection = 100

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
                    #action = model.random_action(s)
                    action = model.action(s, softmax=True)
                else:
                    _time_test = time.time()
                    action = model.action(s, softmax=False)
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
        self.model = model #

    def start(self):
        self.q_data = mp.Queue()
        self.q_data_argmax = mp.Queue()
        self.q_count = mp.Queue()
        self.q_eps = mp.Queue()
        self.q_model = mp.Queue()
        self.q_flag_models = mp.Queue()
        self.q_stat = mp.Queue()

        self.procs = list()
        for idx in range(self.config['learning']['num_processes']):
            target_func = None
            learning_algorithm = self.config['learning']['algorithm']
            if learning_algorithm == 'optimal_q_learning':
                target_func = get_data 
            elif learning_algorithm == 'sarsa':
                target_func = get_data_sarsa

            assert not target_func is None, f"Invalid algorithm is set for learning: {learning_algorithm}"

            proc = mp.Process(
                target=target_func,
                args=(idx, self.config, self.q_data, self.q_data_argmax,
                      self.q_count, self.q_eps, self.q_flag_models,
                      self.q_model)
                )
            proc.start()
            self.procs.append(proc)

    def get_eps(self):
        eps_end = self.config['learning']['eps']['end']
        eps_add = self.config['learning']['eps']['add']
        half_life = self.config['learning']['eps']['half_life']
        eps = eps_end +  eps_add * half_life / (half_life + self.model.step_train)
        return eps

    def get_state_dict_cpu(self):
        state_dict = self.model.state_dict()
        state_dict = { key: state_dict[key].cpu() for key in state_dict }

        return state_dict

    def save_to_replay_buffer(self, size):
        num_data = size
        eps = self.get_eps()
        state_dict_cpu = self.get_state_dict_cpu()

        self.q_count.put(size)
        self.q_eps.put(eps)
        self.q_model.put(state_dict_cpu)
        self.q_flag_models.put( [True for _ in range(self.config['learning']['num_processes'])] )

        num_data_prev = 0
        while num_data > 0:
            if num_data % 100 == 0:
                if num_data != num_data_prev:
                    logger.debug(f"collecting data.. {num_data} are left")
                    print(f"collecting data.. {num_data} are left")
                    num_data_prev = num_data
            if self.q_data.qsize() > 0 :
                sards, score = self.q_data.get()
                self.model.replay_buffer.append(sards, score, 0)
                num_data = num_data - 1
            else:
                time.sleep(1e-3)
        self.q_eps.get()
        self.q_model.get()
        self.q_flag_models.get()

    def terminate(self):
        for proc in self.procs:
            proc.kill()




