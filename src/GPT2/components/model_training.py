import torch
import math
import time
import os
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from GPT2.logging import logger




torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

def train_model(config, train_loader, model, optimizer, raw_model, dist_config):
    # Set up DDP (Distributed Data Parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
    ddp = dist_config.ddp
    ddp_rank = dist_config.ddp_rank
    ddp_local_rank = dist_config.ddp_local_rank
    ddp_world_size = dist_config.ddp_world_size
    master_process = dist_config.master_process
    device = dist_config.device
    device_type = dist_config.device_type
  

    assert config.total_batch_size % (config.B * config.T * ddp_world_size) == 0 , "Make sure total_batch_size is divisible by (B * T * ddp_world_size) "
    grad_accum_step = config.total_batch_size // (config.B * config.T * ddp_world_size)
    if master_process:
        logger.info(f" Total desired batch size: {config.total_batch_size} => calculated gradient accumulation steps: {grad_accum_step}")

    # Just A100 and above: It's working onTensorFloat-32 (TF32) 8x faster 
    # in matrix multipication (nn.Linear Layer). The output of matmul is still FP32
    torch.set_float32_matmul_precision('high')


    # Just A100/V100 and above: Make PyTorch code run faster by compiling PyTorch code into optimized kernels Speedup mainly comes from reducing
    # Python overhead and GPU read/writes,second time we run model with torch.compile is significantly slower than the other runs, 
    # although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
    # and it doesn't work right now on "MPS"

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
            if ddp:
                # We don't want DDP in each micro_step applying all_reduce after calculating the gradient during loss.backward 
                # Since it's just meaningless, we need adding gradient until the last micro_step then applying all_reduce.
                # after running loss.backward(), all the ranks has access to the average of all the gradients
                model.require_backward_grad_sync = (micro_step == grad_accum_step - 1)
            loss.backward()
        if ddp:
            # Since loss_accum outside of DDP cintainer, it hasn't impacted yet and it only shows accumulated loss for the master rank GPU.
            # we want to see the averaged loss of all the processes
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
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
        token_per_sec = (config.B * config.T * grad_accum_step * ddp_world_size) / dt
        if master_process:
            logger.info(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {token_per_sec:.2f}")
    if ddp:
        destroy_process_group()


