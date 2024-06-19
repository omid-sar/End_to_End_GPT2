
from  GPT2.research.gpt2_HF_w_model import GPT, GPTConfig
from GPT2.utils.model_utils import get_device
from GPT2.logging import logger

import torch
from torch.nn import functional as F

device = get_device()
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

"""# ---------------------- Tokenize with the same sentece to compare HF model weights ------------------------
import tiktoken

text = "Hello, I'm a model that can complete sentences. Watch me go!"
num_return_sequences = 5
max_lenght = 50
logger.info(f"Inferencing GPT2 model with HuggingFace GPT2 Weights,[num_return_sequences: {num_return_sequences}],[max_lenght: {max_lenght}], [Sample text: {text}]")

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

model = GPT.from_pretrained('gpt2')
model.to(device)
#model = GPT(GPTConfig()) # if want to try the model with random weights!
model.eval()

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
"""

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
    

train_loader = DataLoaderLite( B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for i in range(50):
    model.train()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")

import sys; sys.exit(0)
#%%

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

