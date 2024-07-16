import torch
import numpy as np
import math
import time
import os
import tiktoken
from pathlib import Path
from torch.nn import functional as F
from torch.distributed import  destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from GPT2.logging import logger
from GPT2.models.gpt2_model import GPT
from GPT2.components.model_evaluation import inference_step, evaluate_hellaswag





def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) 
    ppt = torch.tensor(npt, dtype=torch.long)
    return ppt

class DataLoaderLite:
    def __init__(self, config, dist_config, split):
        self.B = config.B
        self.T = config.T
        self.process_rank = dist_config.ddp_rank
        self.num_processes = dist_config.ddp_world_size
        master_process = dist_config.master_process
        self.split = split
        assert split in {'train', 'val'}
        self.rng = np.random.default_rng(1337)

        data_path = Path(os.path.join(config.local_data_file, config.dataset_name))
        shards = os.listdir(data_path)
        shards = [s for s in shards if split in s ]
        shards = sorted(shards)
        shards = [os.path.join(data_path, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0 , f"No shards found for split: {split}"
        if master_process:
            logger.info(f" Found {len(shards)} shards for solit: {split}")
        self.reset()

    def load_shard(self, filename):
        shard = load_tokens(filename)
        enc = tiktoken.get_encoding("gpt2")
        if self.split == "train":
            # split tokens into documents using the <|endoftext|> token and shuffle
            eot_positions = (torch.where(shard == enc.eot_token)[0] + 1).tolist()
            documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
            self.rng.shuffle(documents)
            shard = torch.cat(documents) # concatenate the documents back together
        return shard


    def reset(self):
        self.current_shard = 0
        if self.split == "train":
            self.rng.shuffle(self.shards)
        self.tokens = self.load_shard(self.shards[self.current_shard])
        #self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position+self.B*self.T+1 ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T * self.num_processes
        # If loading the next batch would be out of bounds, move to the next shard.
        if self.current_position + (self.B * self.T * self.num_processes + 1 )> len(self.tokens):
            self.current_shard += 1
            # reshuffle after each epoch
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_position = self.B * self.T * self.process_rank          
            # self.current_shard = (self.current_shard + 1) % len(self.shards)
            # self.tokens = load_tokens(self.shards[self.current_shard])
            # self.current_position = self.B * self.T * self.process_rank
        return x, y



def train_model(training_config, gpt_config, data_transformation_config, dist_config, use_compile):
    # We need random seed to all model weights initiate the same in all GPUs
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    # Set up DDP (Distributed Data Parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
    ddp = dist_config.ddp
    ddp_rank = dist_config.ddp_rank
    ddp_local_rank = dist_config.ddp_local_rank
    ddp_world_size = dist_config.ddp_world_size
    master_process = dist_config.master_process
    device = dist_config.device
    device_type = dist_config.device_type

    log_file = os.path.join(training_config.traing_log_file, training_config.log_name)
    with open(log_file, "w") as f: # Open for writing to clear the file
        pass
  

    assert training_config.total_batch_size % (training_config.B * training_config.T * ddp_world_size) == 0 , "Make sure total_batch_size is divisible by (B * T * ddp_world_size) "
    grad_accum_step = training_config.total_batch_size // (training_config.B * training_config.T * ddp_world_size)
    if master_process:
        logger.info(f" Total desired batch size: {training_config.total_batch_size} => calculated gradient accumulation steps: {grad_accum_step}")

    train_loader = DataLoaderLite(config=data_transformation_config, dist_config=dist_config, split="train")
    val_loader = DataLoaderLite(config=data_transformation_config, dist_config=dist_config, split="val")
    # Just A100 and above: It's working onTensorFloat-32 (TF32) 8x faster 
    # in matrix multipication (nn.Linear Layer). The output of matmul is still FP32
    torch.set_float32_matmul_precision('high')

    model = GPT(config=gpt_config)
    model.to(device)
    # torch.compile interferes with HellaSwag eval and Generation. there is problem here   
    if use_compile:
        model = torch.compile(model)
    # Just A100/V100 and above: Make PyTorch code run faster by compiling PyTorch code into optimized kernels Speedup mainly comes from reducing
    # Python overhead and GPU read/writes,second time we run model with torch.compile is significantly slower than the other runs, 
    # although it is much faster than the first run. This is because the "reduce-overhead" mode runs a few warm-up iterations for CUDA graphs.
    # and it doesn't work right now on "MPS"
    if ddp:
            model = DDP(model, device_ids=[ddp_local_rank])
    # a consistent way to access the underlying model, whether it's wrapped in DDP or not.
    raw_model = model.module if ddp else model 
    optimizer = raw_model.configure_optimizer(weight_decay=gpt_config.weight_decay, learning_rate=gpt_config.learning_rate, betas=gpt_config.betas ,device_type=device_type)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps 
        if it < training_config.warmup_steps:
            return (it+1) * training_config.max_lr / training_config.warmup_steps
        # 2) if it > lr_decay_iters, return the min_lr
        if it > training_config.max_steps:
            return training_config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - training_config.warmup_steps) / (training_config.max_steps - training_config.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return training_config.min_lr + coeff * (training_config.max_lr-training_config.min_lr)

    for step in range(training_config.max_steps):
        t0 = time.time()
        last_step = (step == training_config.max_steps - 1)

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_step):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # We don't want DDP in each micro_step applying all_reduce after calculating the gradient during loss.backward 
                # Since it's just meaningless, we need adding gradient until the last micro_step then applying all_reduce.
                # after running loss.backward(), all the ranks has access to the average of all the gradients
                model.require_backward_grad_sync = (micro_step == grad_accum_step - 1)
            # Just A100 and above: Automatic Mixed Precision package. It's just applying 
            # in the forward path and it doesn't apply to all layers, just very selective ones
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #%%% torch.autocast doesn't work on other than A100/H100, so the assert doesn't work for other than A100/H10
                logits, loss = model(x,y)
                assert logits.dtype is torch.bfloat16
            loss = loss / grad_accum_step
            loss_accum += loss.detach()
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
        torch.cuda.synchronize() #%%% wait for GPU to finish work
        t1 = time.time()
        dt = (t1- t0)
        token_per_sec = (training_config.B * training_config.T * grad_accum_step * ddp_world_size) / dt
        if master_process:
            logger.info(f"Step {step} | Loss: {loss_accum.item():.6f} | LR: {lr:.4e} | Norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {token_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

        # once in a while, evaluate our validation loss
        if step % training_config.val_steps == 0 or last_step:
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
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #%%% torch.autocast doesn't work on other than A100/H100
                        logits, loss = model(x,y)
                        assert logits.dtype is torch.bfloat16 #%%% torch.autocast doesn't work on other than A100/H100, so the assert doesn't work for other than A100/H10
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            if ddp:
                # Since val_loss_accum outside of DDP cintainer, it hasn't impacted yet and it only shows accumulated loss for the master rank GPU.
                # we want to see the averaged loss of all the processes
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                logger.info(f"Step: {step} | val: {val_loss_accum.item():.4f}\n")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                # Save model checkpoints
                if step > 0 and (step % 5000 == 0 or last_step):
                    model_path = Path(training_config.model_folder) 
                    if not (model_path ).exists():
                        os.makedirs(model_path, exist_ok=True)
                    
                    checkpoint_path = os.path.join(training_config.model_folder, f"model_{step:05d}.pt")
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'val_loss': val_loss_accum.item()
                        }
                    torch.save(checkpoint, checkpoint_path)

        # once in a while evaluate hellaswag
        if ((step > 0 and step % training_config.val_steps == 0 ) or last_step) and (not use_compile):
            master_process, acc_norm = evaluate_hellaswag(model, step, ddp_world_size, ddp_rank, device, device_type, ddp, master_process)
            if master_process:
                with open(log_file, "a") as f:
                    f.write(f"{step} hella {acc_norm:.4f}\n")

        # once in a while, genearte from the model
        if ((step > 0 and step % training_config.val_steps == 0 ) or last_step) and (not use_compile):
            inference_step(model, device, device_type, ddp_rank)
    
    if ddp:
        destroy_process_group()


