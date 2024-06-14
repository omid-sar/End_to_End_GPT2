from dataclasses import dataclass
import torch
import math
import torch.nn as nn
from torch.nn import functional as F


# ----------- Temporary to load Config localy here not main.py -----
from GPT2.config.configuration import ConfigurationManager
import os
os.getcwd()
os.chdir('../../../')
config = ConfigurationManager()
# -------------------------------------------------------------------

@dataclass 
class GPTconfig:
    block_size : int = 1024 # Sequence Length
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd : int = 768 
    mlp_hidden_size: int = 3072


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd is not divisble by n_head" 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
     
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        # NOT really a 'bias', more of a mask, but following HF naming though
        self.register_buffer("bias", torch.tril(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size)

    def forward(self, x):
        # (Batch, Seq_len, embd_dim)
        B, T, C = x.size()
        d_k = self.n_head, C // self.n_head 
        # we concatant Wq, Wk, Wv and multiply them with X (Batch, Seq_len, embd_dim)
        # then slpit then q,k,v and then reshape them based on size of head to calculate MultiHead attention 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (Batch, Seq_len, embd_dim) -> (Batch, Seq_len, n_head, d_k) -> (Batch, n_head, Seq_len, d_k)
        q = q.view(B, T, self.n_head, d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_k).transpose(1, 2)

        att = q @ k.transpose(-2,-1) * (1.0/ math.sqrt(d_k))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0 , float('-inf')) # MASK
        att = F.softmax(att, dim=1)
        #(Batch, n_head, Seq_len, Seq_len) X (Batch, n_head, Seq_len, d_k) = (Batch, n_head, Seq_len, d_k)
        y = att @ v 
        # (Batch, n_head, Seq_len, d_k) -> 
        







class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden_size)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.mlp_hidden_size, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)


class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd)
        )) 
        self.lm_head = nn.Linear(config.vocab_size, config.n_embd, bias=False)