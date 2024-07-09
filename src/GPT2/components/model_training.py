import torch
import tiktoken
import math
import time
import os
from pathlib import Path
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from GPT2.logging import logger




torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

def train_model(config, train_loader, val_loader, model, optimizer, raw_model, dist_config, use_compile):
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
    # torch.compile interferes with HellaSwag eval and Generation. there is problem here   
    if use_compile:
        model = torch.compile(model)

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
        last_step = (step == config.max_steps - 1)

        # once in a while, evaluate our validation loss
        if step % 3 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    # Just A100 and above: Automatic Mixed Precision package. It's just applying 
                    # in the forward path and it doesn't apply to all layers, just very selective ones
                    #***with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
                        #***$$assert logits.dtype is torch.bfloat16
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                # Since val_loss_accum outside of DDP cintainer, it hasn't impacted yet and it only shows accumulated loss for the master rank GPU.
                # we want to see the averaged loss of all the processes
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                logger.info(f"Step: {step} | val: {val_loss_accum.item():.4f}\n")
                # Save model checkpoints
                if step > 0 and (step % 5000 == 0 or last_step):
                    model_path = Path(config.model_folder) 
                    if not (model_path ).exists():
                        os.makedirs(model_path, exist_ok=True)
                    
                    checkpoint_path = os.path.join(config.model_folder, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item()
                        }
                    torch.save(checkpoint, checkpoint_path)

        # once in a while, genearte from the model
        if ((step > 0 and step % 3 == 0 ) or last_step) and (not use_compile):
            model.eval()
            text = "Hello, I'm a model,"
            num_return_sequences = 4
            max_lenght = 32
            # logger.info(f"Inferencing GPT2 model with HuggingFace GPT2 Weights,[num_return_sequences: {num_return_sequences}],[max_lenght: {max_lenght}], [Sample text: {text}]")
            enc = tiktoken.get_encoding('gpt2')
            tokens = enc.encode(text)
            tokens = torch.tensor(tokens, dtype=torch.long)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            while xgen.size(1) < max_lenght:
                # Forward path to create logits
                with torch.no_grad():
                    #****with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x) #(B, T, vocab_size)
                    logits = logits[:,-1,:] # (B, vocab_size) we just take the logits of last token
                    probs = F.softmax(logits, dim=-1) # Get the probabilities (5, vocab_size)
                    # Do top-K sampling of 50 (HF pipeline default)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # topk_probs(5, 50), topk_indices(5, 50)
                    # Select a token from the top-k probabilities 
                    ix = torch.multinomial(topk_probs, 1, generator=sample_rng) #(B, 1)
                    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
                    xgen = torch.cat((xgen, xcol), dim=1)

            # print the generated text
            for i in range(num_return_sequences):
                tokens = xgen[i, :max_lenght].tolist()
                decoded = enc.decode(tokens)
                logger.info(f'{"-" * 75} \n Rank:{ddp_rank} | Sample:{i} | {decoded}')

           
        model.train()
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
            logger.info(f"Step {step} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {token_per_sec:.2f}")
    if ddp:
        destroy_process_group()


