import torch 
import torch.nn as nn
import deepgate as dg

class Transformer_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args 
        self.device = args.device
    
    def forward(self, node_states):
        init_hs, init_hf = node_states
        pass 
        

class DeepGate3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = args.device
        self.tokenizer = dg.Model(args)
        self.tokenizer.load_pretrained()
        self.tokenizer.to(self.device)
        self.encoder = Transformer_Encoder(args)
        
    def forward(self, g):
        # Toknization with DeepGate2  
        hs, hf = self.tokenizer(g)
        init_node_states = (hs, hf)
        
        # Transformer-based Encoder
        node_states = self.encoder(init_node_states)
        