
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from torch import nn
from mamba_ssm import Mamba
from torch_geometric.utils import to_dense_batch
import deepgate as dg 
import torch
import copy
import numpy as np 
from deepgate.arch.mlp import MLP
from deepgate.arch.mlp_aggr import MlpAggr
from deepgate.arch.tfmlp import TFMlpAggr
from deepgate.arch.gcn_conv import AggConv
from torch.nn import LSTM, GRU
def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)
    
    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)
    
    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat([
        node_indices[mask][torch.randperm(mask.sum(), device=identities.device)] for mask in masks
    ])
    return permuted_indices

def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    neighbours = neighbours[idx_select]
    return neighbours

def augment_seq(edge_index, batch, num_k = -1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0]==k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)

def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)
    
    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        
        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]
        
        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)
    
    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices



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
    device = G.gate.device
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

class DeepGate_mamba(dg.Model):
    def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1, num_rounds=4, enable_encode=True, enable_reverse=False):
        dim_hidden = hidden
        super().__init__(num_rounds, dim_hidden, enable_encode, enable_reverse)

        self.args = args
        self.hidden = hidden
        self.record = {}
        self.num_head = attn_heads
        self.max_length = 512
        dim_h = hidden
        self.func_mamba = nn.ModuleList()
        self.stru_mamba = nn.ModuleList()
        self.ff_linear1_hf, self.ff_linear2_hf, self.ff_linear1_hs, self.ff_linear2_hs = nn.ModuleList(),nn.ModuleList(),nn.ModuleList(),nn.ModuleList()
        # self.aggr_and_strc = nn.ModuleList()
        # self.aggr_not_strc = nn.ModuleList()
        # self.aggr_and_func = nn.ModuleList()
        # self.aggr_not_func = nn.ModuleList()
            
        # self.update_and_strc = nn.ModuleList()
        # self.update_and_func = nn.ModuleList()
        # self.update_not_strc = nn.ModuleList()
        # self.update_not_func = nn.ModuleList()
         # configuration
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse        # TODO: enable reverse

        # dimensions
        self.dim_hidden = dim_hidden
        self.dim_mlp = 32

        for i in range(num_rounds):
            self.func_mamba.append(Mamba(d_model=dim_hidden, # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=8,    # Local convolution width
            expand=8,    # Block expansion factor
            ))
            self.stru_mamba.append(Mamba(d_model=dim_hidden, # Model dimension d_model
            d_state=32,  # SSM state expansion factor
            d_conv=8,    # Local convolution width
            expand=8,    # Block expansion factor
            ))
            self.ff_linear1_hf.append(nn.Linear(dim_h, dim_h * 2))
            self.ff_linear2_hf.append(nn.Linear(dim_h * 2, dim_h))
            self.ff_linear1_hs.append(nn.Linear(dim_h, dim_h * 2))
            self.ff_linear2_hs.append(nn.Linear(dim_h * 2, dim_h))

            # Network 
            # self.aggr_and_strc.append( TFMlpAggr(self.dim_hidden*1, self.dim_hidden))
            # self.aggr_not_strc.append( TFMlpAggr(self.dim_hidden*1, self.dim_hidden))
            # self.aggr_and_func.append( TFMlpAggr(self.dim_hidden*2, self.dim_hidden))
            # self.aggr_not_func.append( TFMlpAggr(self.dim_hidden*1, self.dim_hidden))
                
            # self.update_and_strc.append( GRU(self.dim_hidden, self.dim_hidden))
            # self.update_and_func.append( GRU(self.dim_hidden, self.dim_hidden))
            # self.update_not_strc.append( GRU(self.dim_hidden, self.dim_hidden))
            # self.update_not_func.append( GRU(self.dim_hidden, self.dim_hidden))
        #MLP
        # self.ff_linear1_hf = nn.Linear(dim_h, dim_h * 2)
        # self.ff_linear2_hf = nn.Linear(dim_h * 2, dim_h)
        # self.ff_linear1_hs = nn.Linear(dim_h, dim_h * 2)
        # self.ff_linear2_hs = nn.Linear(dim_h * 2, dim_h)
        self.activation = F.relu
        #norm
        self.norm1_local = nn.BatchNorm1d(dim_h)
        self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward_hf(self, batch, hf, hs, round):

        h_out_list = []

        h = hf+hs
        h_in1 = h.clone()

        h_out_list.append(h_in1)

        #GLOBAL MODEL
        h_ind_perm = permute_within_batch(batch.batch)
        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
        h_ind_perm_reverse = torch.argsort(h_ind_perm)

        mod = self.func_mamba[round].to(h_dense.device)
        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]


        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.

        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        
        h = sum(h_out_list)
        h = h + self._ff_block_hf(h,round)

        h = self.norm2(h)

        return h
    
    def forward_hs(self, batch, hf, hs, round):

        h_out_list = []
        h = hf + hs
        h_in1 = h.clone()

        h_out_list.append(h_in1)

        #GLOBAL MODEL

        h_ind_perm = permute_within_batch(batch.batch)
        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
        h_ind_perm_reverse = torch.argsort(h_ind_perm)


        mod = self.stru_mamba[round].to(h_dense.device)
        h_attn = mod(h_dense)[mask][h_ind_perm_reverse]

        h_attn = self.dropout_attn(h_attn)
        h_attn = h_in1 + h_attn  # Residual connection.

        h_attn = self.norm1_attn(h_attn)
        h_out_list.append(h_attn)

        
        h = sum(h_out_list)
        h = h + self._ff_block_hs(h,round)

        h = self.norm2(h)

        return h

    def _ff_block_hf(self, x,round):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_hf[round](x)))
        return self.ff_dropout2(self.ff_linear2_hf[round](x))
    
    def _ff_block_hs(self, x,round):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_hs[round](x)))
        return self.ff_dropout2(self.ff_linear2_hs[round](x))
    
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
            # prob_mask = copy.deepcopy(PI_prob)
            prob_mask = PI_prob
            
        prob_mask = prob_mask.unsqueeze(-1).to(device)
        prob_mask[G.gate != 0] = -1
        hf = prob_mask.expand(num_nodes, self.dim_hidden).clone()
        hf = hf.float()
        
        hs = hs.to(device)
        hf = hf.to(device)
        
        node_state = torch.cat([hs, hf], dim=-1)
        
        and_slices, not_slices, edge_slices = get_slices(G)

        
        # for level in range(1, num_layers_f):
        #     l_and_node = and_slices[level]
        #     if l_and_node.size(0) > 0:
        #         and_edge_index = edge_slices[level]
        #         # Update structure hidden state
        #         msg = self.aggr_and_strc[round](hs, and_edge_index)
        #         and_msg = torch.index_select(msg, dim=0, index=l_and_node)
        #         hs_and = torch.index_select(hs, dim=0, index=l_and_node)
        #         _, hs_and = self.update_and_strc[round](and_msg.unsqueeze(0), hs_and.unsqueeze(0))
        #         hs[l_and_node, :] = hs_and.squeeze(0)
        #         # Update function hidden state
        #         msg = self.aggr_and_func[round](node_state, and_edge_index)
        #         and_msg = torch.index_select(msg, dim=0, index=l_and_node)
        #         hf_and = torch.index_select(hf, dim=0, index=l_and_node)
        #         _, hf_and = self.update_and_func[round](and_msg.unsqueeze(0), hf_and.unsqueeze(0))
        #         hf[l_and_node, :] = hf_and.squeeze(0)

        #     # NOT Gate
        #     l_not_node = not_slices[level]
        #     if l_not_node.size(0) > 0:
        #         not_edge_index = edge_slices[level]
        #         # Update structure hidden state
        #         msg = self.aggr_not_strc[round](hs, not_edge_index)
        #         not_msg = torch.index_select(msg, dim=0, index=l_not_node)
        #         hs_not = torch.index_select(hs, dim=0, index=l_not_node)
        #         _, hs_not = self.update_not_strc[round](not_msg.unsqueeze(0), hs_not.unsqueeze(0))
        #         hs[l_not_node, :] = hs_not.squeeze(0)
        #         # Update function hidden state
        #         msg = self.aggr_not_func[round](hf, not_edge_index)
        #         not_msg = torch.index_select(msg, dim=0, index=l_not_node)
        #         hf_not = torch.index_select(hf, dim=0, index=l_not_node)
        #         _, hf_not = self.update_not_func[round](not_msg.unsqueeze(0), hf_not.unsqueeze(0))
        #         hf[l_not_node, :] = hf_not.squeeze(0)
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
        for round in range(self.num_rounds):
            node_state = torch.cat([hs, hf], dim=-1)
    
            node_embedding = node_state.squeeze(0)
            hs = node_embedding[:, :self.dim_hidden]
            hf = node_embedding[:, self.dim_hidden:]
            hf = self.forward_hf(G, hf, hs, round)
            hs = self.forward_hs(G, hf, hs, round)
            node_state = torch.cat([hs, hf], dim=-1)

        node_embedding = node_state.squeeze(0)
        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        return hs, hf

# class DeepGate_mamba(nn.Sequential):
#     def __init__(self, args, hidden=128, n_layers=12, attn_heads=4, dropout=0.1):
#         super().__init__()
#         self.args = args
#         self.hidden = hidden
#         self.record = {}
#         self.num_head = attn_heads
#         self.max_length = 512
#         dim_h = hidden


#         self.func_mamba = []
#         self.stru_mamba = []

#         for i in range(4):
#             self.func_mamba.append(Mamba(d_model=hidden, # Model dimension d_model
#             d_state=16,  # SSM state expansion factor
#             d_conv=4,    # Local convolution width
#             expand=1,    # Block expansion factor
#             ))
#             self.stru_mamba.append(Mamba(d_model=hidden, # Model dimension d_model
#             d_state=16,  # SSM state expansion factor
#             d_conv=4,    # Local convolution width
#             expand=1,    # Block expansion factor
#             ))
        
#         #MLP
#         self.ff_linear1_hf = nn.Linear(dim_h, dim_h * 2)
#         self.ff_linear2_hf = nn.Linear(dim_h * 2, dim_h)
#         self.ff_linear1_hs = nn.Linear(dim_h, dim_h * 2)
#         self.ff_linear2_hs = nn.Linear(dim_h * 2, dim_h)
#         self.activation = F.relu
#         #norm
#         self.norm1_local = nn.BatchNorm1d(dim_h)
#         self.norm1_attn = nn.BatchNorm1d(dim_h)
#         self.norm2 = nn.BatchNorm1d(dim_h)
#         self.ff_dropout1 = nn.Dropout(dropout)
#         self.ff_dropout2 = nn.Dropout(dropout)
#         self.dropout_attn = nn.Dropout(dropout)

#     def _ff_block_hf(self, x):
#         """Feed Forward block.
#         """
#         x = self.ff_dropout1(self.activation(self.ff_linear1_hf(x)))
#         return self.ff_dropout2(self.ff_linear2_hf(x))
#     def _ff_block_hs(self, x):
#         """Feed Forward block.
#         """
#         x = self.ff_dropout1(self.activation(self.ff_linear1_hs(x)))
#         return self.ff_dropout2(self.ff_linear2_hs(x))
    
#     def forward_hf(self, batch, hf, hs):

#         h = batch.x
#         h_out_list = []

#         h = hf+hs
#         h_in1 = h.clone()

#         h_out_list.append(h_in1)

#         #GLOBAL MODEL
#         mamba_arr_h = []
#         for i in range(5):
#             h_ind_perm = permute_within_batch(batch.batch)
#             h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
#             h_ind_perm_reverse = torch.argsort(h_ind_perm)

#             h_attn_list = []
#             for mod in self.func_mamba:
#                 mod = mod.to(h_dense.device)
#                 h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
#                 h_attn_list.append(h_attn) 
#             h_attn = sum(h_attn_list) / len(h_attn_list)

#             #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
#             mamba_arr_h.append(h_attn)

#         h_attn = sum(mamba_arr_h) / 5
#         h_attn = self.dropout_attn(h_attn)
#         h_attn = h_in1 + h_attn  # Residual connection.

#         h_attn = self.norm1_attn(h_attn)
#         h_out_list.append(h_attn)

        
#         h = sum(h_out_list)
#         h = h + self._ff_block_hf(h)

#         h = self.norm2(h)

#         return h
    
#     def forward_hs(self, batch, hf, hs):

#         h = batch.x
#         h_out_list = []

#         h = hf + hs
#         h_in1 = h.clone()

#         h_out_list.append(h_in1)

#         #GLOBAL MODEL

#         h_ind_perm = permute_within_batch(batch.batch)
#         h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
#         h_ind_perm_reverse = torch.argsort(h_ind_perm)

#         h_attn_list = []
#         for mod in self.func_mamba:
#             mod = mod.to(h_dense.device)
#             h_attn = mod(h_dense)[mask][h_ind_perm_reverse]
#             h_attn_list.append(h_attn) 
#         h_attn = sum(h_attn_list) / len(h_attn_list)

#         h_attn = self.dropout_attn(h_attn)
#         h_attn = h_in1 + h_attn  # Residual connection.

#         h_attn = self.norm1_attn(h_attn)
#         h_out_list.append(h_attn)

        
#         h = sum(h_out_list)
#         h = h + self._ff_block_hf(h)

#         h = self.norm2(h)

#         return h

#     def forward(self, g, hf, hs):
#         bs = g.batch.max().item() + 1
#         hf_tf = self.forward_hf(g, hf, hs)
#         hs_tf = self.forward_hs(g, hf, hs)
#         # for i in range(bs):
#         #     batch_idx = g.forward_index[g.batch==i]
#         #     hf[batch_idx] = hf_tf[i,:batch_idx.shape[0]]
#         #     hs[batch_idx] = hs_tf[i,:batch_idx.shape[0]]
#         return hf_tf, hs_tf