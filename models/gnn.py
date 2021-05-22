''' 
-----------------------------------------------
Explanation:
GNN model to be used for embedding a random graph 
-----------------------------------------------
'''
import random
import numpy as np

import torch
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

class ReplayBuffer:
    def __init__(self, config):
        self.config = config
        self.buffer = []
        self.size = self.config['learning']['size_replay_buffer']

    def reset(self):
        self.buffer = []

    def append(self, sarsa):
        if len(self.buffer) == self.size:
            self.buffer.pop(0)
        self.buffer.append(sarsa)

    def get(self):
        assert len(self.buffer) > self.config['learning']['size_batch'], (
            f"Not enough data is collected for a batch size {size_batch}, currently {len(self.buffer)}"
            )
        return random.sample(self.buffer, self.config['learning']['size_batch'])


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tau = 1.
        self.base_hidden_size = 64
        self.bias = False
        self.sigma = 1e-1

        self.T1 = 5
        self.T2 = 5

        # Used for estimating presence probabilities
        self.fc1_presence = nn.Linear(3, self.base_hidden_size, bias=self.bias)
        self.fc2_presence = nn.Linear(self.base_hidden_size, 1, bias=self.bias)

        self.fc_x_1a = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_embedding_1a = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_l_1a = nn.Linear(self.base_hidden_size, self.base_hidden_size, bias=self.bias)

        self.fc_x_1b = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_embedding_1b = nn.Linear(1, self.base_hidden_size, bias=self.bias)
        self.fc_l_1b = nn.Linear(self.base_hidden_size, self.base_hidden_size, bias=self.bias)

        self.fc_x_2 = nn.Linear(2 * self.base_hidden_size, 2 * self.base_hidden_size, bias=self.bias)
        self.fc_embedding_2 = nn.Linear(1, 2 * self.base_hidden_size, bias=self.bias)
        self.fc_l_2 = nn.Linear(2 * self.base_hidden_size, 2 * self.base_hidden_size, bias=self.bias)

        self.fc_Q = nn.Linear(2 * self.base_hidden_size, 1, bias=False)

        self.replay_buffer = ReplayBuffer(config)

    def forward(self, state, action):
        ''' 
        -----------------------------------------------
        Explanation:
        This function returns Q from state
        -----------------------------------------------
        '''
        assignment = state['assignment_prev'] + action # (n_batch, n_robots, n_nodes)
        x_a = torch.max(state['x_a'] * assignment, dim=1, keepdim=True).values # (n_batch, n_robots, n_nodes)
        x_a = x_a.transpose(1, 2).float() # (n_batch, n_nodes, 1)
        x_b = state['x_b'] # (n_batch, n_nodes, 1)
        edge = state['edge'] # (n_batch, n_cities, n_nodes, 3)

        avail_node_presence = state['avail_node_presence'] # (n_batch, 1, n_nodes)
        avail_node_action = state['avail_node_action'] # (n_batch, 1, n_nodes)
        avail_robot = state['avail_robot'] # (n_batch, 1, n_robots)

        no_robot = torch.sum(avail_robot, dim=-1).squeeze(-1)==0
        assert not no_robot.any(), f"There is no available robot in some batch, {no_robot}"

        assigned_visited_city = (avail_node_presence - assignment) < 0
        assert not assigned_visited_city.any(), f"Robot is assigned to already visited city!, {assigned_visited_city}"

        n_batch = edge.shape[0]
        n_cities = edge.shape[1]
        n_nodes = edge.shape[2]

        u_a = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size)
        u_b = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size)
        gamma = self.sigma * torch.randn(n_batch, n_nodes, 2 * self.base_hidden_size)

        h1_presence = torch.relu(self.fc1_presence(edge))
        h2_presence = torch.relu(self.fc2_presence(h1_presence)).squeeze(-1) # (n_batch, n_cities, n_nodes)
        h2_presence = h2_presence / self.tau
        mask_presence = 1 - torch.eye(n_cities, n_nodes).unsqueeze(0).repeat(n_batch, 1, 1) # eleminate self-feeding presence
        mask_presence = mask_presence * avail_node_presence 
        logit_presence = h2_presence * mask_presence - (1 - mask_presence) * 1e10
        presence_out = torch.softmax(logit_presence, dim = -1)
        presence_in = presence_out.transpose(1, 2) # (n_batch, n_nodes, n_cities)

        edge_dist = edge[:, :, :, 0:1].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1)
        _presence_in = presence_in.unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)

        # First convolution of graphs
        for t in range(self.T1):
            ## Original concating method
            # edge_dist = edge[:, :, :, 0:1].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1), distance between each nodes & cities
            # u_a_rep = u_a.unsqueeze(1).repeat(1, n_nodes, 1, 1)[:, :, :-1, :] # (n_batch. n_nodes, n_cities, self.base_hidden_size)
            # u_a_rep = torch.cat([u_a_rep, edge_dist], dim=-1) # (n_batch. n_nodes, n_cities, self.base_hidden_size + 1)
            
            embedding_dist_1a = torch.sigmoid(self.fc_embedding_1a(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            u_a_rep = u_a.unsqueeze(1).repeat(1, n_nodes, 1, 1)[:, :, :-1, :] # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            u_a_rep = u_a_rep * embedding_dist_1a # (n_batch, n_nodes, n_cities, self.base_hidden_size)

            l_a = torch.matmul(_presence_in, u_a_rep) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_a = l_a.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            u_a = torch.relu(self.fc_l_1a(l_a) + self.fc_x_1a(x_a))

            embedding_dist_1b = torch.sigmoid(self.fc_embedding_1b(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            u_b_rep = u_b.unsqueeze(1).repeat(1, n_nodes, 1, 1)[:, :, :-1, :] # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            u_b_rep = u_b_rep * embedding_dist_1b # (n_batch, n_nodes, n_cities, self.base_hidden_size)

            l_b = torch.matmul(_presence_in, u_b_rep) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_b = l_b.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            u_b = torch.relu(self.fc_l_1b(l_a) + self.fc_x_1b(x_a))

        u_concat = torch.cat([u_a, u_b], dim=-1) # (n_batch, n_nodes, 2 * self.base_hidden_size)
        # Second convolution of graphs
        for t in range(self.T2):
            embedding_dist_2 = torch.sigmoid(self.fc_embedding_2(edge_dist))
            gamma_rep = gamma.unsqueeze(1).repeat(1, n_nodes, 1, 1)[:, :, :-1, :] # (n_batch, n_nodes, n_cities, 2 * self.base_hidden_size)
            gamma_rep = gamma_rep * embedding_dist_2

            l_2 = torch.matmul(_presence_in, gamma_rep) # (n_batch, n_nodes, 1, 2 * self.base_hidden_size)
            l_2 = l_2.squeeze(-2) # (n_batch, n_nodes, 2 * self.base_hidden_size)
            gamma = torch.relu(self.fc_l_2(l_2) + self.fc_x_2(u_concat)) # (n_batch, n_nodes, 2 * self.base_hidden_size)

        Q_utility = self.fc_Q(gamma) # (n_batch, n_nodes, 1)
        Q = torch.sum(Q_utility, dim=1) # (n_batch, 1)

        return Q

    def get_Q_from_list_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float() for key in state
        }
        
        action_numpy = self._convert_list_action_to_numpy(action)
        action_tensor = torch.from_numpy(action_numpy).float()
        Q = self.forward(state_tensor, action_tensor)

        return Q.detach().numpy()

    def get_Q_from_numpy_action(self, state, action):
        state_tensor = {
            key: torch.from_numpy(state[key]).float() for key in state
        }
        action_tensor = torch.from_numpy(action).float()
        Q = self.forward(state_tensor, action)

        return Q.detach().numpy()

    # Action based on learned Q: auctino for multiple robots, argmax for a robot
    def action(self, state):
        # Returns an optimal Q action
        assert state['avail_robot'].shape[0] == 1, (
            f"This function is not designed for batch operation, "
            + f"your trying to use it for batch size {state['avail_robot'].shape[0]}"
            )
        check_multiple_available_robots = np.sum(state['avail_robot']) > 1
        check_multiple_available_nodes = np.sum(state['avail_node_action']) > 1

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes = self._get_idx_avail_nodes_from_state(state)
        
        if ( # multiple robots & multiple nodes
            check_multiple_available_robots
            and check_multiple_available_nodes
            ):
            _action = self._auction(state)
        elif check_multiple_available_nodes: # multiple nodes, one robot
            _action = self._argmax_action(state)
        else: # only one node (base), should go base
            _action = [ None for _ in range(self.config['env']['num_robots']) ]
            for idx_robot in idx_avail_robots:
                _action[idx_robot] = self.config['env']['num_cities'] # go_base

        return _action

    # random action used for epsilon-greedy
    def random_action(self, state):
        # Returns an random action
        # Returns an optimal Q action
        assert state['avail_robot'].shape[0] == 1, (
            f"This function is not designed for batch operation, "
            + f"your trying to use it for batch size {state['avail_robot'].shape[0]}"
            )
        check_multiple_available_robots = np.sum(state['avail_robot']) > 1
        check_multiple_available_nodes = np.sum(state['avail_node_action']) > 1

        idx_avail_robots = self.get_idx_avail_robots_from_state(state)
        idx_avail_nodes = self.get_idx_avail_nodes_from_state(state)

        _action = [None for _ in range(self.config['env']['num_robots'])]
        for idx_robot in idx_avail_robots:
            idx_node = random.sample(idx_avail_nodes, 1)[0]
            _action[idx_robot] = idx_node
            self._remove_node(idx_avail_nodes, idx_node)

        return _action

    def initialize_batch(self):
        self.batch = {
            'assignment_prev': np.zeros([
                self.config['learning']['size_batch'], 
                self.config['env']['num_robots'],
                self.config['env']['num_cities'] + 1
                ]),
            'x_a': np.zeros([
                self.config['learning']['size_batch'], 
                1,
                self.config['env']['num_cities'] + 1
                ]),
        }

    def process_batch(self):
        batch = self.replay_buffer.get()

    def add_to_replay_buffer(self, sarsa):
        self.replay_buffer.append(sarsa)

    def _auction(self, state):
        auction_result = [ None for _ in range(self.config['env']['num_robots']) ]

        idx_avail_robots = self._get_idx_avail_robots_from_state(state)
        idx_avail_nodes = self._get_idx_avail_nodes_from_state(state)

        final_auction_action = np.zeros([1, self.config['env']['num_robots'], self.config['env']['num_cities'] + 1])
        updated_avail_node_action = state['avail_node_action']

        n_for_auction = len(idx_avail_robots)
        for _ in range(n_for_auction):
            # Duplicate state for computing Qs for every possible nodes for a robot
            _state = {
                key: np.tile(state[key], [len(idx_avail_nodes), 1, 1]) if len(state[key].shape)==3
                else np.tile(state[key], [len(idx_avail_nodes), 1, 1, 1])
                for key in state
                }

            # Make actions for computing Qs for every possible nodes for a robot
            action_numpys = [ 
                np.zeros([
                    len(idx_avail_nodes), 
                    self.config['env']['num_robots'], 
                    self.config['env']['num_cities'] + 1
                    ]) + final_auction_action
                for _ in range(len(idx_avail_robots))
                ]

            idx_optimal_avail_nodes = []
            optimal_Qs = []

            # Set hypothetical actions for each robots and nodes
            for j, idx_robot in enumerate(idx_avail_robots):
                for k, idx_node in enumerate(idx_avail_nodes):
                    action_numpys[j][k, idx_robot, idx_node] = 1

            count = 0
            for action_numpy in action_numpys:
                Q_avail_nodes_of_a_robot = self.get_Q_from_numpy_action(_state, action_numpy).reshape(-1)
                idx_optimal_avail_node = Q_avail_nodes_of_a_robot.argmax()
                optimal_Q = Q_avail_nodes_of_a_robot[idx_optimal_avail_node]

                idx_optimal_avail_nodes.append(idx_optimal_avail_node)
                optimal_Qs.append(optimal_Q)

                logger.info(f"Q_avail_nodes_of_a_robot {idx_avail_robots[count]}: {Q_avail_nodes_of_a_robot}")
                count += 1

            logger.info(f"optimal_Qs: {optimal_Qs}")

            idx_optimal_avail_robot = np.array(optimal_Qs).argmax()
            argmax_robot = idx_avail_robots[idx_optimal_avail_robot]
            argmax_node = idx_avail_nodes[idx_optimal_avail_nodes[idx_optimal_avail_robot]]

            logger.info(f"argmax_robot, argmax_node: {argmax_robot, argmax_node}")
            logger.info(f"idx_avail_nodes: {idx_avail_nodes}")

            final_auction_action[0, argmax_robot, argmax_node] = 1
            updated_avail_node_action[0, 0, argmax_node] = 0
            
            self._remove_node(idx_avail_nodes, argmax_node)
            idx_avail_robots.remove(argmax_robot)

            auction_result[argmax_robot] = argmax_node

        return auction_result

    # Argmax action when there is only one available robot
    def _argmax_action(self, state):
        _action = [None for _ in range(self.config['env']['num_robots'])]

        idx_avail_robots = self.get_idx_avail_robots_from_state(state)
        idx_avail_nodes = self.get_idx_avail_nodes_from_state(state)

        assert len(idx_avail_robots)==1, "argmax action can not be computed when there is multiple available robots"
        idx_robot = idx_avail_robots[0]

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

        Q_avail_nodes_of_a_robot = self.get_Q_from_numpy_action(_state, action_numpy).reshape(-1)
        idx_optimal_avail_node = Q_avail_nodes_of_a_robot.argmax()
        argmax_node = idx_avail_nodes[idx_optimal_avail_node]

        logger.info(f"Q_avail_nodes: {Q_avail_nodes_of_a_robot}")
        logger.info(f"idx_avail_nodes: {idx_avail_nodes}")

        _action[idx_robot] = argmax_node

        return _action

    def _remove_node(self, idx_nodes, idx_node):
        if not idx_node == self.config['env']['num_cities']:
            idx_nodes.remove(idx_node) # We do not exclude base for avail_node_action

    def _get_idx_avail_robots_from_state(self, state):
        avail_robots = state['avail_robot'][0, 0, :] # (n_robots)
        idx_avail_robots = np.arange(self.config['env']['num_robots'])
        mask_avail_robots = avail_robots > 0
        idx_avail_robots = idx_avail_robots[mask_avail_robots].tolist()

        return idx_avail_robots

    def _get_idx_avail_nodes_from_state(self, state):
        avail_node_action = state['avail_node_action'][0, 0, :] #(n_cities + 1)
        idx_avail_nodes = np.arange(self.config['env']['num_cities'] + 1)
        mask_avail_nodes = avail_node_action > 0
        idx_avail_nodes = idx_avail_nodes[mask_avail_nodes].tolist()

        return idx_avail_nodes

    def _convert_list_action_to_numpy(self, action):
        # convert action list into a tensor
        action_numpy = np.zeros([1, self.config['env']['num_robots'], self.config['env']['num_cities'] + 1])
        for idx, _action in enumerate(action):
            if _action is None:
                continue
            else:
                action_numpy[0, idx, _action] = 1

        return action_numpy













