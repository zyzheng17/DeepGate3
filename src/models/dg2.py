import deepgate as dg 
import torch
import copy

class DeepGate2(dg.Model):
    def __init__(self, num_rounds=1, dim_hidden=128, enable_encode=True, enable_reverse=False):
        super().__init__(num_rounds, dim_hidden, enable_encode, enable_reverse)
    
    def forward(self, G, PI_prob=None):
        device = next(self.parameters()).device
        num_nodes = len(G.gate)
        num_layers_f = max(G.forward_level).item() + 1
        num_layers_b = max(G.backward_level).item() + 1
        
        # initialize the structure hidden state
        if self.enable_encode:
            hs = torch.zeros(num_nodes, self.dim_hidden)
            hs, max_sim = dg.utils.generate_hs_init(G, hs, self.dim_hidden)
        else:
            hs = torch.zeros(num_nodes, self.dim_hidden)
            max_sim = 0
        
        # initialize the function hidden state
        # prob_mask = copy.deepcopy(G.prob)
        if PI_prob is None:
            prob_mask = [0.5] * len(G.gate)
            prob_mask = torch.tensor(prob_mask).unsqueeze(1)
        else:
            prob_mask = copy.deepcopy(PI_prob)
        prob_mask = prob_mask.to(device)
        prob_mask[G.gate != 0] = -1
        hf = prob_mask.expand(num_nodes, self.dim_hidden).clone()
        hf = hf.float()
        
        hs = hs.to(device)
        hf = hf.to(device)
        
        edge_index = G.edge_index

        node_state = torch.cat([hs, hf], dim=-1)
        and_mask = G.gate == 1
        not_mask = G.gate == 2

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                # forward layer
                layer_mask = G.forward_level == level

                # AND Gate
                l_and_node = G.forward_index[layer_mask & and_mask]
                if l_and_node.size(0) > 0:
                    and_edge_index, and_edge_attr = dg.dag_utils.subgraph(l_and_node, edge_index, dim=1)
                    
                    # Update structure hidden state
                    msg = self.aggr_and_strc(hs, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))
                    hs[l_and_node, :] = hs_and.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_and_func(node_state, and_edge_index, and_edge_attr)
                    and_msg = torch.index_select(msg, dim=0, index=l_and_node)
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0)

                # NOT Gate
                l_not_node = G.forward_index[layer_mask & not_mask]
                if l_not_node.size(0) > 0:
                    not_edge_index, not_edge_attr = dg.dag_utils.subgraph(l_not_node, edge_index, dim=1)
                    # Update structure hidden state
                    msg = self.aggr_not_strc(hs, not_edge_index, not_edge_attr)
                    not_msg = torch.index_select(msg, dim=0, index=l_not_node)
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0)
                    # Update function hidden state
                    msg = self.aggr_not_func(hf, not_edge_index, not_edge_attr)
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