from dataclasses import dataclass
import torch
import inspect
import math
import torch.nn as nn
from torch.nn import functional as F
from GPT2.logging import logger

@dataclass 
class GPTConfig:
    block_size : int = 1024 # Sequence Length
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd : int = 768 

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0, "n_embd is not divisble by n_head" 
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
     
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        # Adding kind of flag to the modules to cancel out standard deviation growth inside the residual
        # Stearms in each layer (we have two layers in each block: attention and MLP) 
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # NOT really a 'bias', more of a mask, but following HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # (Batch, Seq_len, embd_dim)
        B, T, C = x.size()
        d_k = C // self.n_head 
        # we concatant Wq, Wk, Wv and multiply them with X (Batch, Seq_len, embd_dim)
        # then slpit then q,k,v and then reshape them based on size of head to calculate MultiHead attention 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # (Batch, Seq_len, embd_dim) -> (Batch, Seq_len, n_head, d_k) -> (Batch, n_head, Seq_len, d_k)
        q = q.view(B, T, self.n_head, d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, d_k).transpose(1, 2)

        # att = q @ k.transpose(-2,-1) * (1.0/ math.sqrt(d_k))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0 , float('-inf')) # MASK
        # att = F.softmax(att, dim=-1)
        # #(Batch, n_head, Seq_len, Seq_len) X (Batch, n_head, Seq_len, d_k) = (Batch, n_head, Seq_len, d_k)
        # y = att @ v 
        # # (Batch, n_head, Seq_len, d_k) -> (Batch, Seq_len, n_head, d_k) -> (Batch, Seq_len, embd_dim)
        y = F.scaled_dot_product_attention(q, k , v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y 
        

class MLP(nn.Module):

    def __init__(self, config):
        super(MLP, self).__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # Adding kind of flag to the modules to cancel out standard deviation growth inside the residual
        # Stearms in each layer (we have two layers in each block: attention and MLP) 
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


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

    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
            ln_f = nn.LayerNorm(config.n_embd)
        )) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight

        # init params

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f" The forward Sequence length is {T} which CANNOT be longer that block size {self.config.block_size}"
        pos =torch.arange(0, T , dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # Positional embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T , n_embd)
        x = pos_emb + tok_emb

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        logger.info(f"loading weights from pretrained gpt {model_type}")

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        logger.info(f"Successfully weights loaded from pretrained gpt {model_type}")
        return model
    def configure_optimizer(self, weight_decay, learning_rate, device_type):
        # Start with all of Parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()} # have to change to self
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Create optim groups. Any parameters that is 2D will be decayed, otherwise No.
        #  i.e. all weight tensors in matmuls + embeddings decay, all biases and layersnorms Don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"Number of Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it available 
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        logger.info(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

"""    

model = GPT(GPTConfig())
weight_decay = []


# Start with all of Parameters (that require grad)
param_dict = {pn: p for pn, p in model.named_parameters()} # have to change to self
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
# Create optim groups. Any parameters that is 2D will be decayed, otherwise No.
#  i.e. all weight tensors in matmuls + embeddings decay, all biases and layersnorms Don't
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 ]
optim_group = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0}
]
num_decay_params = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
logger.info(f"Number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
logger.info(f"Number of Non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
# Create AdamW optimizer and use the fused version if it available 
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == "cuda"
logger.info(f"using fused AdamW: {use_fused}")

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

inspect.signature(torch.optim.AdamW)
torch.optim.AdamW().pa
inspect.signature(torch.optim.AdamW).parameters"""