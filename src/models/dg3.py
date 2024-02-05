import torch 
import deepgate as dg
import torch.nn as nn 
from .pool import PoolNet
from .mlp import MLP

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.deepgate = dg.Model(args)
        self.deepgate.load_pretrained()
        self.deepgate.freeze()
        self.deepgate.to(args.device)
        # Pooling 
        self.stru_pool = PoolNet(args)
        self.func_pool = PoolNet(args)
        # Fast Simulation (Function Supervision)
        self.func_net = MLP(
            dim_in=args.dim_hidden * args.no_pi + args.dim_hidden,
            dim_hidden=args.mlp_hidden, 
            dim_pred=1,
            num_layer=args.mlp_layer,
            norm_layer=args.norm_layer,
            act_layer=args.act_layer,
            sigmoid=True
        )
        # PPA Prediction (Structure Supervision)
        self.stru_net = MLP(
            dim_in=args.dim_hidden,
            dim_hidden=args.mlp_hidden, 
            dim_pred=2,
            num_layer=args.mlp_layer,
            norm_layer=args.norm_layer,
            act_layer=args.act_layer,
        )
        
    def forward(self, g):
        hs, hf = self.deepgate(g)
        hs_states = torch.cat([hs[g.PIs], hs[g.POs]], dim=-1)
        hf_states = torch.cat([hf[g.PIs], hf[g.POs]], dim=-1)
        hs_graph = self.stru_pool(hs_states)
        hf_graph = self.func_pool(hf_states)
        
        # Fast Simulation
        fastsim_inputs = torch.cat([[g.pi_workload] * self.args.dim_hidden, hf_graph], dim=-1)
        fastsim_output = self.func_net(fastsim_inputs)
        
        # PPA Prediction
        ppa_inputs = hs_graph
        ppa_output = self.stru_net(ppa_inputs)
        
        return hs_graph, hf_graph, fastsim_output, ppa_output
        