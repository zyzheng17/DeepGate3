import deepgate as dg 
import torch
import copy
import numpy as np 

def generate_orthogonal_vectors(n, dim):
    if n < dim * 8:
        # Choice 1: Generate n random orthogonal vectors in R^dim
        # Generate an initial random vector
        v0 = np.random.randn(dim)
        v0 /= np.linalg.norm(v0)
        # Generate n-1 additional vectors
        vectors = [v0]
        for i in range(n-1):
            while True:
                # Generate a random vector
                v = np.random.randn(dim)

                # Project the vector onto the subspace spanned by the previous vectors
                for j in range(i+1):
                    v -= np.dot(v, vectors[j]) * vectors[j]

                if np.linalg.norm(v) > 0:
                    # Normalize the vector
                    v /= np.linalg.norm(v)
                    break

            # Append the vector to the list
            vectors.append(v)
    else: 
        # Choice 2: Generate n random vectors:
        vectors = np.random.rand(n, dim) - 0.5
        for i in range(n):
            vectors[i] = vectors[i] / np.linalg.norm(vectors[i])

    return vectors

def generate_hs_init(G, hs, no_dim):
    if G.batch == None:
        batch_size = 1
    else:
        batch_size = G.batch.max().item() + 1
    for batch_idx in range(batch_size):
        if G.batch == None:
            pi_mask = (G.forward_level == 0)
        else:
            pi_mask = (G.batch == batch_idx) & (G.forward_level == 0)
        pi_node = G.forward_index[pi_mask]
        pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim)
        hs[pi_node] = torch.tensor(np.array(pi_vec), dtype=torch.float)
    
    return hs, -1

def get_slices(G):
    device = G.x.device
    edge_index = G.edge_index
    
    # Edge slices 
    edge_level = torch.index_select(G.forward_level, dim=0, index=edge_index[1])
    # sort edge according to level
    edge_indices = torch.argsort(edge_level)
    edge_index = edge_index[:, edge_indices]
    edge_level_cnt = torch.bincount(edge_level).tolist()

    edge_index_slices = torch.split(edge_index, list(edge_level_cnt), dim=1)
    
    # Index slices
    and_index_slices = []
    not_index_slices = []
    and_mask = (G.gate == 1).squeeze(1)
    not_mask = (G.gate == 2).squeeze(1)
    for level in range(0, torch.max(G.forward_level).item() + 1):
        and_level_nodes = torch.nonzero((G.forward_level == level) & and_mask).squeeze(1)
        not_level_nodes = torch.nonzero((G.forward_level == level) & not_mask).squeeze(1)
        and_index_slices.append(and_level_nodes)
        not_index_slices.append(not_level_nodes)
    
    return and_index_slices, not_index_slices, edge_index_slices

class DeepGate2(dg.Model):
    def __init__(self, num_rounds=1, dim_hidden=128, enable_encode=True, enable_reverse=False):
        super().__init__(num_rounds, dim_hidden, enable_encode, enable_reverse)
    
    def forward(self, G, PI_prob=None):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        num_layers_f = torch.max(G.forward_level).item() + 1
        
        # initialize the structure hidden state
        if self.enable_encode:
            hs = torch.zeros(num_nodes, self.dim_hidden)
            hs, _ = generate_hs_init(G, hs, self.dim_hidden)
        else:
            hs = torch.zeros(num_nodes, self.dim_hidden)
        
        # initialize the function hidden state
        # prob_mask = copy.deepcopy(G.prob)
        if PI_prob is None:
            prob_mask = [0.5] * len(G.gate)
            prob_mask = torch.tensor(prob_mask).unsqueeze(1)
        else:
            prob_mask = copy.deepcopy(PI_prob)
        prob_mask = prob_mask.unsqueeze(-1).to(device)
        prob_mask[G.gate != 0] = -1
        hf = prob_mask.expand(num_nodes, self.dim_hidden).clone()
        hf = hf.float()
        
        hs = hs.to(device)
        hf = hf.to(device)
        
        node_state = torch.cat([hs, hf], dim=-1)
        
        and_slices, not_slices, edge_slices = get_slices(G)

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                l_and_node = and_slices[level]
                if l_and_node.size(0) > 0:
                    and_edge_index = edge_slices[level]
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate
                l_not_node = not_slices[level]
                if l_not_node.size(0) > 0:
                    not_edge_index = edge_slices[level]
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_not_func(hf, not_edge_index)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0)
                
                # Update node state
                node_state = torch.cat([hs, hf], dim=-1)

        node_embedding = node_state.squeeze(0)
        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        return hs, hf