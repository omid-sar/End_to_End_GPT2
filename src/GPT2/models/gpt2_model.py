from dataclasses import dataclass
import torch
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
    block_size : int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd : int = 768 

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        self.config = config



class Block(nn.Module):

    def __init__(self, config):
        super(Block, self).__init__()
        self.config = config

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        



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