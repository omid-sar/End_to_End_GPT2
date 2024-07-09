import tiktoken
import torch
from GPT2.logging import logger
from torch.nn import functional as F


def inference_step(model, device, ddp_rank):
        model.eval()
        text = "Hello, I'm a model,"
        num_return_sequences = 4
        max_length = 32
        
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        
        while xgen.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            logger.info(f'{"-" * 75} \n Rank:{ddp_rank} | Sample:{i} | {decoded}')