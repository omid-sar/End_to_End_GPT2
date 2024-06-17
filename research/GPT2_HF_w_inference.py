from  GPT2.models.gpt2_model import GPT
from GPT2.logging import logger

import torch
from torch.nn import functional as F

# --------------------------------- Load weights from HG to our local --------------------------
text = "Hello, I'm a model that can complete sentences. Watch me go!"
num_return_sequences = 5
max_lenght = 50

model = GPT.from_pretrained('gpt2')
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)



import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)
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


