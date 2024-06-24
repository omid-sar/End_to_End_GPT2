
from  GPT2.research.gpt2_HF_w_model import GPT, GPTConfig
from GPT2.utils.model_utils import get_device
from GPT2.logging import logger

import torch
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
        print(f"loaded {len(self.tokens)} tokens")
        print(f"loaded {len(self.tokens) // (B*T)} batches")

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

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(10):
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
    optimizer.step()
    #***torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1- t0)*1000
    token_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms, tok/sec: {token_per_sec:.2f}")

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

