import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding
import sys
# sys.path.append('/uac/gds/zyzheng23/projects/deepgate3/src/bert_model/embedding')
# from token import TokenEmbedding
from .embedding.position import PositionalEmbedding
from .embedding.segment import SegmentEmbedding

import torch

class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size=None, hidden=768, n_layers=12, attn_heads=4, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        # self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)
        # self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.mask_graph_token = nn.Parameter(torch.zeros([1,hidden]))
        self.mask_PO_token = nn.Parameter(torch.zeros([1,hidden]))
        self.position = PositionalEmbedding(d_model=hidden)
        self.segment = SegmentEmbedding(embed_size=hidden)
        self.dropout = nn.Dropout(p=dropout)
        # multi-layers transformer blocks, deep network
        self.transformer_enc = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        self.transformer_dec = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
        
        self.graph_mlp = nn.Sequential(nn.Linear(hidden, hidden*4),
                            nn.ReLU(),
                            nn.LayerNorm(hidden*4),
                            nn.Linear(hidden*4, hidden))
        
        self.PO_mlp = nn.Sequential(nn.Linear(hidden, hidden*4),
                            nn.ReLU(),
                            nn.LayerNorm(hidden*4),
                            nn.Linear(hidden*4, hidden))
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict_ = checkpoint['state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = self.state_dict()
        
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape))
                    state_dict[k] = model_state_dict[k]
            else:
                print('Drop parameter {}.'.format(k))
        for k in model_state_dict:
            if not (k in state_dict):
                print('No param {}.'.format(k))
                state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)

    def forward(self, PI, PO, segment_info):
        
        # # first mask graph and predict grap
        # f = torch.cat([torch.stack([self.mask_graph_token]*PI.shape[0]),PI,PO],dim=1) # (1+6+1) * 128
        # # x =  f + self.position(f)
        # x =  f + self.position(f) + self.segment(segment_info)
        # # x = self.dropout(x)
        # for transformer in self.transformer_enc:
        #     x = transformer.forward(x, mask=None) # fix length -> no padded token -> no mask

        G_emb = self.graph_mlp(PO).unsqueeze(1)

        #then mask PO
        f = torch.cat([G_emb,PI,torch.stack([self.mask_PO_token]*PI.shape[0])],dim=1) # (1+6+1) * 128
        # x =  f + self.position(f)
        x =  f + self.position(f) + self.segment(segment_info)
        # x = self.dropout(x)
        for transformer in self.transformer_dec:
            x = transformer.forward(x, mask=None) # fix length -> no padded token -> no mask
        PO_emb = self.PO_mlp(x[:,-1,:]).unsqueeze(1)

        return G_emb,PO_emb



    def forward_original(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
    
