
from  GPT2.research.gpt2_HF_w_model import GPT, GPTConfig
from GPT2.utils.model_utils import get_device
from GPT2.logging import logger

import torch
import math
import time
from torch.nn import functional as F

device = get_device()
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


# ---------------------- Tokenizing very first tiny_shakespeare and create batch ------------------------

import tiktoken

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open ("tiny_shakespeare.txt", "r") as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        logger.info(f"loaded {len(self.tokens)} tokens")
        logger.info(f"loaded {len(self.tokens) // (B*T)} batches")

        self.current_position = 0

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position+self.B*self.T+1 ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T
        if self.current_position > len(self.tokens):
            self.current_position = 0 
        return x, y

# -------------------------------------- Training --------------------------------------   

train_loader = DataLoaderLite(B=4, T=1024)
# Just A100 and above: It's working onTensorFloat-32 (TF32) 8x faster 
# in matrix multipication (nn.Linear Layer). The output of matmul is still FP32
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)
# Just A100/V100 and above: Make PyTorch code run faster by compiling PyTorch code into optimized kernels Speedup mainly comes from reducing
# Python overhead and GPU read/writes,second time we run model with torch.compile is significantly slower than the other runs, 
# although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
#***model = torch.compile(model)
max_lr = 6e-4
min_lr =max_lr * 0.1
warmup_steps = 5
max_steps = 10
def get_lr(it):
    # 1) linear warmup for warmup_iters steps 
    if it < warmup_steps:
        return (it+1) * max_lr / warmup_steps
    # 2) if it > lr_decay_iters, return the min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr-min_lr)

optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # Just A100 and above: Automatic Mixed Precision package. It's just applying 
    # in the forward path and it doesn't apply to all layers, just very selective ones
    #***with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x,y)
        #***$$assert logits.dtype is torch.bfloat16
    #import code; code.interact(local=locals())
    loss.backward()
    # GLOBAL NORM = 1(computes a single norm over all the gradients of the parameters in the model, not individually for each layer or block)
    #  Clipping by norm preserves the direction of the gradient vector but reduces its magnitude.
    # cliping by norm less likely to interfere with natural convergence (prevent gradient shocks for an abnormal batch of data) of learning algorithms compare value clipping 
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    # determine and set the learning rate for this iteration
    lr =get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #***torch.cuda.synchronize() # wait for GPU to finish work
    t1 = time.time()
    dt = (t1- t0)*1000
    token_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    logger.info(f"step {step} | loss: {loss.item():.6f} | lr: {lr:.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {token_per_sec:.2f}")

import sys; sys.exit(0)

# --------------------------------- Generate the next token  --------------------------
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_lenght:
    # Forward path to create logits
    with torch.no_grad():
        logits = model(x) #(B, T, vocab_size)
        logits = logits[:,-1,:] # (B, vocab_size) we just take the logits of last token
        probs = F.softmax(logits, dim=-1) # Get the probabilities (5, vocab_size)
        # Do top-K sampling of 50 (HF pipeline default)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # topk_probs(5, 50), topk_indices(5, 50)
        # Select a token from the top-k probabilities 
        ix = torch.multinomial(topk_probs, 1) #(B, 1)
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        x = torch.cat((x, xcol), dim=1)


for i in range(num_return_sequences):
    tokens = x[i, :max_lenght].tolist()
    decode = enc.decode(tokens)
    print(f'{"*" * 50} \n {decode}')

