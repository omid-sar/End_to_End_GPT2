
from  GPT2.models.gpt2_model import GPT, GPTConfig
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

def train_model(config, train_loader, model, optimizer):

    assert config.total_batch_size % (config.B * config.T) == 0 , "Make sure total_batch_size is divisible by (B * T) "
    grad_accum_step = config.total_batch_size // (config.B * config.T)
    logger.info(f" Total desired batch size: {config.total_batch_size} => calculated gradient accumulation steps: {grad_accum_step}")

    # Just A100 and above: It's working onTensorFloat-32 (TF32) 8x faster 
    # in matrix multipication (nn.Linear Layer). The output of matmul is still FP32
    torch.set_float32_matmul_precision('high')

    # Just A100/V100 and above: Make PyTorch code run faster by compiling PyTorch code into optimized kernels Speedup mainly comes from reducing
    # Python overhead and GPU read/writes,second time we run model with torch.compile is significantly slower than the other runs, 
    # although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
    # and it doesn't work right now on "MPS"
    #***model = torch.compile(model)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps 
        if it < config.warmup_steps:
            return (it+1) * config.max_lr / config.warmup_steps
        # 2) if it > lr_decay_iters, return the min_lr
        if it > config.max_steps:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.max_lr-config.min_lr)

    for step in range(config.max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_step):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # Just A100 and above: Automatic Mixed Precision package. It's just applying 
            # in the forward path and it doesn't apply to all layers, just very selective ones
            #***with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
                #***$$assert logits.dtype is torch.bfloat16
            #import code; code.interact(local=locals())
            loss = loss / grad_accum_step
            loss_accum += loss.detach()
            loss.backward()
        # GLOBAL NORM = 1(computes a single norm over all the gradients of the parameters in the model, 
        # not individually for each layer or block. Clipping by norm preserves the direction of the gradient vector 
        # but reduces its magnitude. cliping by norm less likely to interfere with natural convergence 
        # (prevent gradient shocks for an abnormal batch of data) of learning algorithms compare value clipping 
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # determine and set the learning rate for this iteration
        lr =get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        #***torch.cuda.synchronize() # wait for GPU to finish work
        t1 = time.time()
        dt = (t1- t0)
        token_per_sec = (config.B * config.T * grad_accum_step) / dt
        logger.info(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {token_per_sec:.2f}")


