import os
import sys
sys.path.append(os.path.abspath( os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn

from models.gnn import Model as M


class Model(M):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln_u = nn.LayerNorm(self.base_hidden_size)
        #self.ln_embedding_1 = nn.LayerNorm(self.base_hidden_size)

        self.ln_gamma = nn.LayerNorm(self.base_hidden_size)
        #self.ln_embedding_2 = nn.LayerNorm(self.base_hidden_size)

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
        #presence_prev = state['presence_prev']

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

        x = torch.cat(
            [
                #x_a[:, :, self.config['env']['num_robots']:self.config['env']['num_robots'] + 1], 
                x_a, 
                x_b, 
                coord, 
                avail_node_presence.transpose(-2,-1)
                ],
             dim=-1
             ) # (n_batch, n_nodes, 3)
        
        u = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size).to(self.device)
        gamma = self.sigma * torch.randn(n_batch, n_nodes, self.base_hidden_size).to(self.device)

        h0_presence = torch.cat(
            [
            edge, 
            x_a[:, :-1, :].unsqueeze(-2).repeat(1, 1, n_nodes, 1), 
            x_a.unsqueeze(-3).repeat(1, n_cities, 1, 1)
            ],
            dim = -1
            )
        h1_presence = self.activation(self.fc1_presence(h0_presence))
        h2_presence = self.fc2_presence(h1_presence).squeeze(-1) # (n_batch, n_cities, n_nodes)
        h2_presence = h2_presence / self.tau
        mask_presence = 1 - torch.eye(n_cities, n_nodes).unsqueeze(0).repeat(n_batch, 1, 1).to(self.device) # eleminate self-feeding presence
        mask_presence = mask_presence * avail_node_presence 
        mask_presence[:, :, -1] = 0.0 # mask out base node when calculating presence_out
        logit_presence = h2_presence * mask_presence - (1 - mask_presence) * 1e10
        presence_out = torch.softmax(logit_presence, dim = -1)  # (n_batch, n_cities, n_nodes)
        
        # # handling visited nodes
        if not 'presence_prev' in self.config['learning'] or not self.config['learning']['presence_prev']:
            presence_out = avail_node_presence[:, :, :-1].transpose(-2, -1) * presence_out # masking presence out from visited nodes
        else:
            mask_drawed_presence_out = torch.sum(presence_prev, dim=-1) > 0
            mask_drawed_presence_out = mask_drawed_presence_out.float().unsqueeze(-1)
            presence_out = mask_drawed_presence_out * presence_prev + (1 - mask_drawed_presence_out) * presence_out
        
        if self.show_presence:
            print(f"presnce_out: {presence_out[0].max(dim=-1)}")

        presence_in = presence_out.transpose(1, 2).unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)
        edge_dist = edge[:, :, :, 0:self.dim_edge].transpose(1, 2) # (n_batch, n_nodes, n_cities, 1)
        #_presence_in = presence_in.unsqueeze(-2) # (n_batch, n_nodes, 1, n_cities)

        # First convolution of graphs
        for t in range(self.T1):
            ## Original concating method            
            embedding_dist_1 = torch.tanh(self.fc_embedding_1(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_1 = u[:, :-1, :].unsqueeze(1) * embedding_dist_1 # (n_batch, 1 -> n_nodes, n_cities, self.base_hidden_size)

            l_1 = torch.matmul(presence_in, embedding_dist_1) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_1 = l_1.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            #l_a = torch.matmul(presence_in, u_a[:, :-1, :])
            u = u + self.activation(self.fc_l_1(l_1) + self.fc_x_1(x) )

        del l_1, x, x_a, x_b
        # Second convolution of graphs
        for t in range(self.T2):
            embedding_dist_2 = torch.tanh(self.fc_embedding_2(edge_dist)) # (n_batch, n_nodes, n_cities, self.base_hidden_size)
            embedding_dist_2 = gamma[:, :-1, :].unsqueeze(1) * embedding_dist_2 # (n_batch, n_nodes, n_cities, self.base_hidden_size)

            l_2 = torch.matmul(presence_in, embedding_dist_2) # (n_batch, n_nodes, 1, self.base_hidden_size)
            l_2 = l_2.squeeze(-2) # (n_batch, n_nodes, self.base_hidden_size)
            #l_2 = torch.matmul(presence_in, gamma[:, :-1, :])
            gamma = gamma + self.activation(self.fc_l_2(l_2) + self.fc_x_2(u)) # (n_batch, n_nodes, self.base_hidden_size)
        del u, l_2

        sum_gamma_remained = torch.sum(gamma * avail_node_presence.transpose(-2, -1), dim=-2) # (n_batch, self.base_hidden_size)
        sum_gamma_done = torch.sum(gamma * (1 - avail_node_presence.transpose(-2, -1)), dim=-2) # (n_batch, self.base_hidden_size)
        cat_gamma = torch.cat([sum_gamma_remained, sum_gamma_done], dim=-1) # (n_batch, 2 * self.base_hidden_size)
        #Q = self.fc_Q(cat_gamma) # (n_batch, 1)
        Q = self.fc_Q(sum_gamma_remained) # (n_batch, 1)

        return Q
