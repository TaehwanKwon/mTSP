import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn

from models.gnn import Model as M


class Model(M):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc_pred = nn.Linear(self.base_hidden_size, self.config['env']['num_robots'], bias=self.bias)
        self.fc_embedding_pred = nn.Linear(self.config['env']['num_robots'], self.base_hidden_size, bias=self.bias)

    def forward(self, state, action):
        ''' 
        -----------------------------------------------
        Explanation:
        This function returns Q from state
        -----------------------------------------------
        '''
        assignment = state['assignment_prev'] + action # (n_batch, n_robots, n_nodes)
        x_a = torch.sum(state['x_a'] * assignment.unsqueeze(-1), dim=1) # (n_batch, n_nodes, d)
        x_b = state['x_b'] # (n_batch, n_nodes, 1)
        coord = state['coord'] # (n_batch, n_nodes, 2)
        edge = state['edge'] # (n_batch, n_cities, n_nodes, 3)
        presence_prev = state['presence_prev']

        avail_node_presence = state['avail_node_presence'] # (n_batch, 1, n_nodes)

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
            mask_drawed_presence_out = mask_drawed_presence_out.float().unsqueeze(-1)
            presence_out = mask_drawed_presence_out * presence_prev + (1 - mask_drawed_presence_out) * presence_out
        
        presence_in = presence_out.transpose(1, 2).unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)
        #edge_dist = edge[:, :, :, 0:1].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1)
        edge_dist = edge.transpose(1, 2) # (n_batch, n_nodes, n_cities, 1)

        # First convolution of graphs
        for t in range(self.T1):
            ## Original concating method
            
            #embedding_dist_1 = self.ln_embedding_1( torch.tanh( self.fc_embedding_1(edge_dist)) ) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_1 = torch.tanh( self.fc_embedding_1(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_1 = u[:, :-1, :].unsqueeze(1) * embedding_dist_1 # (n_batch, 1 -> n_nodes, n_cities, self.base_hidden_size)

            l_1 = torch.matmul(presence_in, embedding_dist_1) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_1 = l_1.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            u = torch.relu(self.fc_l_1(l_1) + self.fc_x_1(x))

        del l_1, x, x_a, x_b

        pred = torch.softmax(self.fc_pred(u), dim=-1) # (n_batch, n_nodes, n_robots)

        # Second convolution of graphs
        for t in range(self.T2):
            #embedding_dist_2 = self.ln_embedding_2( torch.tanh(self.fc_embedding_2(edge_dist)) )# (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_2 = torch.tanh( self.fc_embedding_2(edge_dist) )# (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_2 = gamma[:, :-1, :].unsqueeze(1) * embedding_dist_2 # (n_batch, n_nodes, n_cities, self.base_hidden_size)

            l_2 = torch.matmul(presence_in, embedding_dist_2) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_2 = l_2.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size
            gamma = torch.relu(
                self.fc_l_2(l_2)
                + self.fc_x_2(u)
                + self.fc_embedding_pred(pred)
                ) # (n_batch, n_nodes, self.base_hidden_size)
        del u, l_2

        sum_gamma_remained = torch.sum(gamma * avail_node_presence.transpose(-2, -1), dim=-2) # (n_batch, self.base_hidden_size)
        sum_gamma_done = torch.sum(gamma * (1 - avail_node_presence.transpose(-2, -1)), dim=-2) # (n_batch, self.base_hidden_size)
        cat_gamma = torch.cat([sum_gamma_remained, sum_gamma_done], dim=-1) # (n_batch, 2 * self.base_hidden_size)
        Q = self.fc_Q(cat_gamma) # (n_batch, 1)

        return Q, pred

    def get_Q_from_list_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float().to(self.device) for key in state if not (key=='avail_node_action' and key=='avail_robot')
        }
        
        action_numpy = self._convert_list_action_to_numpy(action)
        action_tensor = torch.from_numpy(action_numpy).float().to(self.device)
        with torch.no_grad():
            Q, _ = self.forward(state_tensor, action_tensor)
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
            Q, _ = self.forward(state_tensor, action_tensor)
        del state_tensor
        del action_tensor
        Q = Q.detach().cpu().numpy()

        return Q

    def get_pred_from_numpy_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float().to(self.device) for key in state if not (key=='avail_node_action' and key=='avail_robot')
        }
        action_tensor = torch.from_numpy(action).float().to(self.device)
        with torch.no_grad():
            _, pred = self.forward(state_tensor, action_tensor)
        del state_tensor
        del action_tensor
        pred = pred.detach().cpu().numpy()

        return pred

    def get_Q_from_tensor_action(self, state, action):
        Q, _ = self.forward(state, action)
        return Q

    def _set_batch(self):
        batch = self.replay_buffer.sample()
        is_optimal_q_learning = len(batch[0]) == 5
        is_sarsa = len(batch[0]) == 6
        if is_optimal_q_learning:
            state_tuple, action_tuple, reward_tuple, done_tuple, state_next_tuple = zip(*batch)
        elif is_sarsa:
            state_tuple, action_tuple, reward_tuple, done_tuple, state_next_tuple, action_next_tuple = zip(*batch)
        
        #self.sync_models()
        self.load_target()

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
            self.batch['state_final'][idx] = state_next_tuple[idx]['state_final'][0]

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
            'state_final': torch.from_numpy(self.batch['state_final']).float().to(self.device),
        }

        return batch

    def get_processed_batch(self):
        batch = self._get_batch()
        Q_next_max = self.get_Q_from_tensor_action(batch['state_next'], batch['argmax_action']).detach() # We do not generate gradient from target Q value

        self.load_current()
        Q, pred = self.forward(batch['state'], batch['action'])

        processed_batch = dict()
        processed_batch['Q'] = Q
        processed_batch['reward'] = batch['reward']
        processed_batch['done'] = batch['done']
        processed_batch['Q_next_max'] = Q_next_max
        processed_batch['pred'] = pred
        processed_batch['state_final'] = batch['state_final']

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
                    3 + self.config['env']['num_robots']
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
                    4
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
            'state_final': (
                self.config['learning']['size_batch'], 
                self.config['env']['num_cities'] + 1,
                self.config['env']['num_robots']
                )
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
            'state_final': np.zeros( [*self.shapes['state_final']] ),
            }
