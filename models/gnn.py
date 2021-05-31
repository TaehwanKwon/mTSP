''' 
-----------------------------------------------
Explanation:
GNN model to be used for embedding a random graph 
-----------------------------------------------
'''
import time
import copy
import random

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

import logging
logger = logging.getLogger(__name__)

from threading import Thread

def _get_argmax_action(model, done_tuple, state_next_tuple, action):

    argmax_action_list = []
    for idx, state_next in enumerate(state_next_tuple):
        if not done_tuple[idx]:
            argmax_action = model.action(state_next)
        else:
            argmax_action = {'numpy': np.zeros([ *action.shape ])} # add an unused dummy action if the game is done
        argmax_action_list.append(argmax_action['numpy'])
    
    return argmax_action_list

class ReplayBuffer:
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.buffer = []
        self.size = self.config['learning']['size_replay_buffer']

    def reset(self):
        self.buffer = []

    def append(self, data):
        # ('state, action, reward, state_next, done')
        if len(self.buffer) == self.size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample(self):
        ''' 
        ############ CAUTION ############
        random.sample() does not copy data, it takes reference. 
        Do not change the data after sampling.
        '''
        assert len(self.buffer) >= self.config['learning']['size_batch'], (
            f"Not enough data is collected for a batch size {self.config['learning']['size_batch']}, "
            + f"currently {len(self.buffer)}"
            )
        return random.sample(self.buffer, self.config['learning']['size_batch'])

class Model(nn.Module):
    def __init__(self, config, device='cpu', extra_gpus=None):
        super().__init__()
        self.config = config
        self.tau = 2.
        self.base_hidden_size = 128
        self.bias = False
        self.sigma = 1e-3

        self.T1 = 4
        self.T2 = 4

        # Used for estimating presence probabilities
        self.fc1_presence = nn.Linear(3, self.base_hidden_size, bias=self.bias)
        self.fc2_presence = nn.Linear(self.base_hidden_size, 1, bias=self.bias)

        self.fc_x_1 = nn.Linear(7, self.base_hidden_size, bias=self.bias)
        self.fc_embedding_1 = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_l_1 = nn.Linear(self.base_hidden_size, self.base_hidden_size, bias=self.bias)

        self.fc_x_2 = nn.Linear(self.base_hidden_size, self.base_hidden_size, bias=self.bias)
        self.fc_embedding_2 = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_l_2 = nn.Linear(self.base_hidden_size, self.base_hidden_size, bias=self.bias)

        self.fc_Q = nn.Linear(2 * self.base_hidden_size, 1)

        self.replay_buffer = ReplayBuffer(config)
        self.device = device
        self.step_train = 0
        self.tau_softmax = 0.5

        self.extra_gpus = extra_gpus

    def set_extra_gpus(self):
        self.model_list = [self]
        if self.extra_gpus:
            self.model_list.extend( [Model(self.config, device=idx_gpu).to(idx_gpu) for idx_gpu in self.extra_gpus] )

    def sync_models(self):
        state_dict = self.state_dict()
        def _sync_model(model, state_dict):
            _state_dict = {key: state_dict[key].to(model.device) for key in state_dict}
            model.load_state_dict(_state_dict)

        threads = list()
        for model in self.model_list[1:]:
            thread = Thread(target=_sync_model, args=(model, state_dict))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def forward(self, state, action):
        ''' 
        -----------------------------------------------
        Explanation:
        This function returns Q from state
        -----------------------------------------------
        '''
        assignment = state['assignment_prev'] + action # (n_batch, n_robots, n_nodes)
        x_a = torch.sum(state['x_a'] * assignment.unsqueeze(-1), dim=1) # (n_batch, n_nodes, 3)
        x_b = state['x_b'] # (n_batch, n_nodes, 1)
        coord = state['coord'] # (n_batch, n_nodes, 2)
        edge = state['edge'] # (n_batch, n_cities, n_nodes, 3)
        presence_prev = state['presence_prev']

        avail_node_presence = state['avail_node_presence'] # (n_batch, 1, n_nodes)
        #avail_node_action = state['avail_node_action'] # (n_batch, 1, n_nodes)
        
        #avail_robot = state['avail_robot'] # (n_batch, 1, n_robots)
        #no_robot = torch.sum(avail_robot, dim=-1).squeeze(-1)==0
        #if no_robot.any(): logger.debug(f"There is no available robot in some batch, {no_robot}, may be the game is done")

        assigned_visited_city = (avail_node_presence - assignment) < 0
        assert not assigned_visited_city.any(), f"Robot is assigned to already visited city!, {assigned_visited_city}"

        n_batch = edge.shape[0]
        n_cities = edge.shape[1]
        n_nodes = edge.shape[2]

        x = torch.cat([x_a, x_b, coord, avail_node_presence.transpose(-2,-1)], dim=-1) # (n_batch, n_nodes, 3)
        u = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size).to(self.device)
        gamma = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size).to(self.device)

        h1_presence = torch.relu(self.fc1_presence(edge))
        h2_presence = torch.relu(self.fc2_presence(h1_presence)).squeeze(-1) # (n_batch, n_cities, n_nodes)
        h2_presence = h2_presence / self.tau
        mask_presence = 1 - torch.eye(n_cities, n_nodes).unsqueeze(0).repeat(n_batch, 1, 1).to(self.device) # eleminate self-feeding presence
        mask_presence = mask_presence * avail_node_presence 
        logit_presence = h2_presence * mask_presence - (1 - mask_presence) * 1e10
        presence_out = torch.softmax(logit_presence, dim = -1)  # (n_batch, n_cities, n_nodes)
        
        # handling visited nodes
        if not 'presence_prev' in self.config['learning'] or not self.config['learning']['presence_prev']:
            presence_out = avail_node_presence[:, :, :-1].transpose(-2, -1) * presence_out # masking presence out from visited nodes
        else:
            mask_drawed_presence_out = torch.sum(presence_prev, dim=-1) > 0
            presence_out[mask_drawed_presence_out] = presence_prev[mask_drawed_presence_out]
        
        presence_in = presence_out.transpose(1, 2).unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)

        edge_dist = edge[:, :, :, 0:1].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1)
        #_presence_in = presence_in.unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)

        # First convolution of graphs
        for t in range(self.T1):
            ## Original concating method
            # edge_dist = edge[:, :, :, 0:1].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1), distance between each nodes & cities
            # u_a_rep = u_a.unsqueeze(1).repeat(1, n_nodes, 1, 1)[:, :, :-1, :] # (n_batch. n_nodes, n_cities, self.base_hidden_size)
            # u_a_rep = torch.cat([u_a_rep, edge_dist], dim=-1) # (n_batch. n_nodes, n_cities, self.base_hidden_size + 1)
            
            embedding_dist_1 = torch.tanh(self.fc_embedding_1(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_1 = u[:, :-1, :].unsqueeze(1) * embedding_dist_1 # (n_batch, 1 -> n_nodes, n_cities, self.base_hidden_size)

            l_1 = torch.matmul(presence_in, embedding_dist_1) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_1 = l_1.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            #l_a = torch.matmul(presence_in, u_a[:, :-1, :])
            u = torch.relu(self.fc_l_1(l_1) + self.fc_x_1(x) )

        del l_1, x, x_a, x_b
        # Second convolution of graphs
        for t in range(self.T2):
            embedding_dist_2 = torch.tanh(self.fc_embedding_2(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_2 = gamma[:, :-1, :].unsqueeze(1) * embedding_dist_2 # (n_batch, n_nodes, n_cities, self.base_hidden_size)

            l_2 = torch.matmul(presence_in, embedding_dist_2) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_2 = l_2.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            #l_2 = torch.matmul(presence_in, gamma[:, :-1, :])
            gamma = torch.relu(self.fc_l_2(l_2) + self.fc_x_2(u)) # (n_batch, n_nodes, self.base_hidden_size)
        del u, l_2

        sum_gamma_remained = torch.sum(gamma * avail_node_presence.transpose(-2, -1), dim=-2) # (n_batch, self.base_hidden_size)
        sum_gamma_done = torch.sum(gamma * (1 - avail_node_presence.transpose(-2, -1)), dim=-2) # (n_batch, self.base_hidden_size)
        cat_gamma = torch.cat([sum_gamma_remained, sum_gamma_done], dim=-1) # (n_batch, 2 * self.base_hidden_size)
        Q = self.fc_Q(cat_gamma) # (n_batch, 1)

        return Q

    def get_Q_from_list_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float().to(self.device) for key in state if not (key=='avail_node_action' and key=='avail_robot')
        }
        
        action_numpy = self._convert_list_action_to_numpy(action)
        action_tensor = torch.from_numpy(action_numpy).float().to(self.device)
        with torch.no_grad():
            Q = self.forward(state_tensor, action_tensor)
        del state_tensor
        del action_tensor
        Q = Q.detach().cpu().numpy()

        return Q

    def get_Q_from_numpy_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float().to(self.device) for key in state if not (key=='avail_node_action' and key=='avail_robot')
        }
        action_tensor = torch.from_numpy(action).float().to(self.device)
        with torch.no_grad():
            Q = self.forward(state_tensor, action_tensor)
        del state_tensor
        del action_tensor
        Q = Q.detach().cpu().numpy()

        return Q

    def get_Q_from_tensor_action(self, state, action):
        Q = self.forward(state, action)
        return Q

    # Action based on learned Q: auctino for multiple robots, argmax for a robot
    def action(self, state, softmax=True):
        # Returns an optimal Q action
        assert state['avail_robot'].shape[0] == 1, (
            f"This function is not designed for batch operation, "
            + f"your trying to use it for batch size {state['avail_robot'].shape[0]}"
            )
        check_multiple_available_robots = np.sum(state['avail_robot']) > 1
        check_multiple_available_nodes = np.sum(state['avail_node_action'][0, :, :-1]) > 0

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes_list = self._get_idx_avail_nodes_list_from_state(state)

        # print(f"idx_avail_nodes_list: {idx_avail_nodes_list}")
        # print(f"avail_node_action: {state['avail_node_action']}")
        if ( # multiple robots & multiple nodes
            check_multiple_available_robots
            and check_multiple_available_nodes
            ):
            action_list = self._auction(state, softmax=softmax)
        elif check_multiple_available_nodes: # multiple nodes, one robot
            action_list = self._argmax_action(state, softmax=softmax)
        else: # only one node (base), should go base
            action_list = [ None for _ in range(self.config['env']['num_robots']) ]
            for idx_robot in idx_avail_robots:
                action_list[idx_robot] = self.config['env']['num_cities'] # go_base

        action_numpy = self._convert_list_action_to_numpy(action_list)

        # print(f":action_list: {action_list}")

        return {
            'list': action_list,
            'numpy': action_numpy,
        }

    # random action used for epsilon-greedy
    def random_action(self, state):
        # Returns an random action
        # Returns an optimal Q action
        assert state['avail_robot'].shape[0] == 1, (
            f"This function is not designed for batch operation, "
            + f"your trying to use it for batch size {state['avail_robot'].shape[0]}"
            )
        check_multiple_available_robots = np.sum(state['avail_robot']) > 1
        check_multiple_available_nodes = np.sum(state['avail_node_action'][0, :, :-1]) > 0

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes_list = self._get_idx_avail_nodes_list_from_state(state)

        chosen_nodes = []
        action_list = [None for _ in range(self.config['env']['num_robots'])]
        for idx_robot in idx_avail_robots:
            idx_avail_nodes = idx_avail_nodes_list[idx_robot]

            # remove already chosen nodes
            for chosen_node in chosen_nodes:
                if not chosen_node == self.config['env']['num_cities'] and chosen_node in idx_avail_nodes:
                    idx_avail_nodes.remove(chosen_node)

            idx_node = random.sample(idx_avail_nodes, 1)[0]
            action_list[idx_robot] = int(idx_node)
            self._remove_node(idx_avail_nodes, idx_node)

            chosen_nodes.append(idx_node)

        action_numpy = self._convert_list_action_to_numpy(action_list)

        return {
            'list': action_list,
            'numpy': action_numpy,
        }

    def _set_batch(self):
        batch = self.replay_buffer.sample()
        is_optimal_q_learning = len(batch[0]) == 5
        is_sarsa = len(batch[0]) == 6
        if is_optimal_q_learning:
            state_tuple, action_tuple, reward_tuple, done_tuple, state_next_tuple = zip(*batch)
        elif is_sarsa:
            state_tuple, action_tuple, reward_tuple, done_tuple, state_next_tuple, action_next_tuple = zip(*batch)
        
        self.sync_models()

        assert len(state_next_tuple) % self.config['learning']['num_processes'] == 0, 'Currently we only support batch size proportional to number of total gpus'
        m = len(state_next_tuple) // self.config['learning']['num_processes']
        for idx_data in range(self.config['learning']['num_processes']):
            _done_tuple = done_tuple[m * idx_data: m * (idx_data + 1)]
            _state_next_tuple = state_next_tuple[m * idx_data: m * (idx_data + 1)]
            q_data_argmax = (_done_tuple, _state_next_tuple, action_tuple[0])
            self.simulator.q_data_argmax.put( (idx_data, q_data_argmax) )
        
        argmax_action_numpy_list = [None for _ in range(self.config['learning']['num_processes'])]
        for _ in range(self.config['learning']['num_processes']):
            idx_data, argmax_action_numpy = self.simulator.q_data.get()
            argmax_action_numpy_list[idx_data] = argmax_action_numpy
            
        _argmax_action_numpy_list = list()
        for argmax_action_numpy in argmax_action_numpy_list:
            _argmax_action_numpy_list += argmax_action_numpy
        argmax_action_numpy_list = _argmax_action_numpy_list

        for idx in range(self.config['learning']['size_batch']):
            for key in self.batch['state']:
                self.batch['state'][key][idx] = state_tuple[idx][key][0]
            self.batch['action'][idx] = action_tuple[idx]
            self.batch['reward'][idx, 0] = reward_tuple[idx]
            self.batch['done'][idx, 0] = float(done_tuple[idx])
            for key in self.batch['state_next']:
                self.batch['state_next'][key][idx] = state_next_tuple[idx][key][0]
            self.batch['argmax_action'][idx] = argmax_action_numpy_list[idx][0]

    def _get_batch(self):
        self._set_batch()
        batch = {
            'state': {
                key: torch.from_numpy(self.batch['state'][key]).float().to(self.device) for key in self.batch['state']
            },
            'action': torch.from_numpy(self.batch['action']).float().to(self.device),
            'reward': torch.from_numpy(self.batch['reward']).float().to(self.device),
            'done': torch.from_numpy(self.batch['done']).float().to(self.device),
            'state_next': {
                key: torch.from_numpy(self.batch['state_next'][key]).float().to(self.device) for key in self.batch['state_next']
            },
            'argmax_action': torch.from_numpy(self.batch['argmax_action']).float().to(self.device),
        }

        return batch

    def get_processed_batch(self):
        batch = self._get_batch()
        Q = self.get_Q_from_tensor_action(batch['state'], batch['action'])
        Q_next_max = self.get_Q_from_tensor_action(batch['state_next'], batch['argmax_action']).detach() # We do not generate gradient from target Q value

        processed_batch = dict()
        processed_batch['Q'] = Q
        processed_batch['reward'] = batch['reward']
        processed_batch['done'] = batch['done']
        processed_batch['Q_next_max'] = Q_next_max

        return processed_batch

    def initialize_batch(self):
        self.shapes = {
            'state':{
                'assignment_prev': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_robots'],
                    self.config['env']['num_cities'] + 1
                    ),
                'presence_prev': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_cities'], # no '+1' since there is no edge going out from the base
                    self.config['env']['num_cities'] + 1
                    ),
                'x_a': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_robots'],
                    self.config['env']['num_cities'] + 1,
                    3
                    ),
                'x_b': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_cities'] + 1,
                    1
                    ),
                'coord': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_cities'] + 1,
                    2
                    ),
                'edge': (
                    self.config['learning']['size_batch'], 
                    self.config['env']['num_cities'], # no '+1' since there is no edge going out from the base
                    self.config['env']['num_cities'] + 1, 
                    3
                    ),
                'avail_node_presence': (
                    self.config['learning']['size_batch'], 
                    1,
                    self.config['env']['num_cities'] + 1
                    ),
                },
            'action': (
                self.config['learning']['size_batch'], 
                self.config['env']['num_robots'],
                self.config['env']['num_cities'] + 1
                ),
            'reward': (
                self.config['learning']['size_batch'], 
                1
                ),
            'done': (
                self.config['learning']['size_batch'], 
                1
                ),
        }
        self.batch = {
            'state': {
                key: np.zeros([ *self.shapes['state'][key] ]) for key in self.shapes['state']
                },
            'action': np.zeros([ *self.shapes['action'] ]),
            'reward': np.zeros([ *self.shapes['reward'] ]),
            'done': np.zeros([ *self.shapes['done'] ]),
            'state_next':{
                key: np.zeros([ *self.shapes['state'][key] ]) for key in self.shapes['state']
                },
            'argmax_action': np.zeros([ *self.shapes['action'] ]),
            }

    def add_to_replay_buffer(self, data):
        self.replay_buffer.append(data)

    def _auction(self, state, softmax=False):
        auction_result = [ None for _ in range(self.config['env']['num_robots']) ]

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes_list = self._get_idx_avail_nodes_list_from_state(state)
        max_len_avail_nodes = [len(idx_avail_nodes) for idx_avail_nodes in idx_avail_nodes_list]

        final_auction_action = np.zeros([1, self.config['env']['num_robots'], self.config['env']['num_cities'] + 1])
        updated_avail_node_action = state['avail_node_action']

        chosen_nodes = []
        n_for_auction = len(idx_avail_robots)
        for _ in range(n_for_auction):

            idx_optimal_avail_nodes = []
            optimal_Qs = []

            count = 0
            for i, idx_robot in enumerate(idx_avail_robots):
                idx_avail_nodes = idx_avail_nodes_list[idx_robot]
                # remove already chosen nodes
                for chosen_node in chosen_nodes:
                    if not chosen_node == self.config['env']['num_cities'] and chosen_node in idx_avail_nodes:
                        idx_avail_nodes.remove(chosen_node)

                # Duplicate state for computing Qs for every possible nodes for a robot
                _state = {
                        key: np.tile(state[key], [len(idx_avail_nodes), 1, 1]) if len(state[key].shape)==3
                        else np.tile(state[key], [len(idx_avail_nodes), 1, 1, 1])
                        for key in state
                        }
                
                # Make actions for computing Qs for every possible nodes for a robot
                action_numpy = np.zeros([
                                    len(idx_avail_nodes), 
                                    self.config['env']['num_robots'], 
                                    self.config['env']['num_cities'] + 1
                                    ]) + final_auction_action

                # Set hypothetical actions for each robots and nodes
                for j, idx_node in enumerate(idx_avail_nodes):
                    action_numpy[j, idx_robot, idx_node] = 1

                Q_avail_nodes_of_a_robot = self.get_Q_from_numpy_action(_state, action_numpy).reshape(-1)
                idx_optimal_avail_node = Q_avail_nodes_of_a_robot.argmax()
                optimal_Q = Q_avail_nodes_of_a_robot[idx_optimal_avail_node]

                idx_optimal_avail_nodes.append(idx_avail_nodes[idx_optimal_avail_node])
                optimal_Qs.append(optimal_Q)

                #logger.debug(f"Q_avail_nodes_of_a_robot {idx_avail_robots[count]}: {Q_avail_nodes_of_a_robot}")
                count += 1

            #logger.debug(f"optimal_Qs: {optimal_Qs}")
            optimal_Qs = np.array(optimal_Qs)
            idx_optimal_avail_robot = optimal_Qs.argmax()
            argmax_robot = idx_avail_robots[idx_optimal_avail_robot]
            argmax_node = idx_optimal_avail_nodes[idx_optimal_avail_robot]
            chosen_nodes.append(argmax_node)

            #logger.debug(f"argmax_robot, argmax_node: {argmax_robot, argmax_node}")
            #logger.debug(f"idx_avail_nodes: {idx_avail_nodes}")

            final_auction_action[0, argmax_robot, argmax_node] = 1
            updated_avail_node_action[0, 0, argmax_node] = 0
            
            self._remove_node(idx_avail_nodes, argmax_node)
            idx_avail_robots.remove(argmax_robot)

            auction_result[argmax_robot] = argmax_node

        return auction_result

    # Argmax action when there is only one available robot
    def _argmax_action(self, state, softmax=False):
        action_list = [None for _ in range(self.config['env']['num_robots'])]

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes_list = self._get_idx_avail_nodes_list_from_state(state)

        assert len(idx_avail_robots)==1, "argmax action can not be computed when there is multiple available robots"
        idx_robot = idx_avail_robots[0]
        idx_avail_nodes = idx_avail_nodes_list[idx_robot]

        _state = {
            key: np.tile(state[key], [len(idx_avail_nodes), 1, 1]) if len(state[key].shape)==3
            else np.tile(state[key], [len(idx_avail_nodes), 1, 1, 1])
            for key in state
            }

        action_numpy = np.zeros([
            len(idx_avail_nodes), 
            self.config['env']['num_robots'], 
            self.config['env']['num_cities'] + 1
            ])

        for i, idx_node in enumerate(idx_avail_nodes):
            action_numpy[i, idx_robot, idx_node] = 1
        _time_test = time.time()
        Q_avail_nodes_of_a_robot = self.get_Q_from_numpy_action(_state, action_numpy).reshape(-1)
        #time_test = time.time() - _time_test; print(f"time_test_forward: {time_test}")
        Q_avg = np.mean(Q_avail_nodes_of_a_robot)
        std = np.sum( (Q_avail_nodes_of_a_robot - Q_avg) ** 2) ** 0.5 + 1e-3
        self.p = (
            np.exp( (Q_avail_nodes_of_a_robot - Q_avg) / std)
            / np.sum( np.exp( (Q_avail_nodes_of_a_robot - Q_avg) / std) )
            )
        if softmax:
            idx_optimal_avail_node = np.random.choice(len(idx_avail_nodes), 1, p=self.p)[0]
        else:
            idx_optimal_avail_node = Q_avail_nodes_of_a_robot.argmax()
        argmax_node = idx_avail_nodes[idx_optimal_avail_node]

        #logger.debug(f"Q_avail_nodes: {Q_avail_nodes_of_a_robot}")
        #logger.debug(f"idx_avail_nodes: {idx_avail_nodes}")

        action_list[idx_robot] = int(argmax_node)

        return action_list

    def _remove_node(self, idx_nodes, idx_node):
        if not idx_node == self.config['env']['num_cities']:
            idx_nodes.remove(idx_node) # We do not exclude base for avail_node_action

    def _get_idx_avail_robots_from_state(self, state):
        avail_robots = state['avail_robot'][0, 0, :] # (n_robots)
        idx_avail_robots = np.arange(self.config['env']['num_robots'])
        mask_avail_robots = avail_robots > 0
        idx_avail_robots = idx_avail_robots[mask_avail_robots].tolist()

        return idx_avail_robots

    def _get_idx_avail_nodes_list_from_state(self, state):
        idx_avail_nodes_list = []
        avail_node_action = state['avail_node_action'][0, :, :] #(n_robots, n_cities + 1)
        idx_avail_nodes = np.arange(self.config['env']['num_cities'] + 1).reshape(1, -1)

        masked_idx_nodes = avail_node_action * idx_avail_nodes
        mask_avail_nodes = avail_node_action > 0

        for i in range(self.config['env']['num_robots']):
            idx_avail_nodes_list.append(
                np.int32(masked_idx_nodes[i][mask_avail_nodes[i]]).tolist()
                )        

        return idx_avail_nodes_list

    def _convert_list_action_to_numpy(self, action_list):
        # convert action list into a tensor
        action_numpy = np.zeros([1, self.config['env']['num_robots'], self.config['env']['num_cities'] + 1])
        for idx, _action in enumerate(action_list):
            if _action is None:
                continue
            else:
                assert int(_action) == _action, "int(_action) and _action should be same."
                action_numpy[0, idx, int(_action)] = 1

        return action_numpy













