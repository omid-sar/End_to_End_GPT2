import tiktoken
import torch
from GPT2.logging import logger
from torch.nn import functional as F
from GPT2.benchmarks.hellaswag import iterate_examples, render_example
import torch.distributed as dist


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

def evaluate_hellaswag(model, step, ddp_world_size, ddp_rank, device, device_type, ddp, master_process):
        try:
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                # render the example into tokens and labels
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)
                # get the logits
                with torch.no_grad():
                    #***with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            # reduce the stats across all processes
            if ddp:
                num_total = torch.tensor(num_total, dtype=torch.long, device=device)
                num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
                dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
                num_total = num_total.item()
                num_correct_norm = num_correct_norm.item()
            acc_norm = num_correct_norm / num_total
            if master_process:
                logger.info(f" Step:{step} | HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
        
def get_most_likely_row(tokens, mask, logits):
    try:
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)
        if torch.isnan(avg_loss).any():
            logger.warning("NaN values detected in avg_loss")
            return None
        if avg_loss.numel() == 0:
            logger.warning("avg_loss is empty")
            return None

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm
    except Exception as e:
        logger.error(f"Error in get_most_likely_row: {str(e)}")
        return None
