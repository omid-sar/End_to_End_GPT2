"""import os
import torch
import torch.nn as nn
from pathlib import Path

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from GPT2.logging import logger
from GPT2.utils.common import create_directories
from GPT2.utils.model_utils import get_weights_file_path, latest_weights_file_path
from GPT2.components.model_evaluation import run_validation
"""

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


total_batch_size = 8192 #***524299 # 2^19 based on 124M GPT3 mdoel
B = 4 # Micro batch size
T = 1024 # Sequence length
assert total_batch_size % (B * T) ==0 , "Make sure total_batch_size is divisible by (B * T) "
grad_accum_step = total_batch_size // (B * T)
logger.info(f" Total desired batch size: {total_batch_size} => calculated gradient accumulation steps: {grad_accum_step}")

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
warmup_steps = 2
max_steps = 4
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
    dt = (t1- t0)
    token_per_sec = (train_loader.B * train_loader.T * grad_accum_step) / dt
    logger.info(f"step {step} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.2f} sec | tok/sec: {token_per_sec:.2f}")

import sys; sys.exit(0)



def train_model(config, train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, model, device):


    create_directories([config.model_folder])
    model = model.to(device)

    writer = SummaryWriter(config.tensorboard_log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr)


    initial_epoch = 0
    global_step = 0
    preload =config.preload

    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        logger.info(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        logger.info(f"No model to preload, starting from scratch")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)


    for epoch in range(initial_epoch, config.num_epochs):
            torch.cuda.empty_cache()
            model.train()
            batch_iterator = tqdm(train_dataloader, desc=f" Processing epoch {epoch:02d}")
            for batch in batch_iterator:
                model.train()

                encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
                decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
                encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
                decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            
                encoder_output = model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
                decoder_output = model.decode(encoder_output, decoder_input, decoder_mask, encoder_mask) #(B, seq_len, d_model)
                proj_output = model.project(decoder_output) #(B, seq_len, tgt_vocab_size)

                # Compare the output with the label
                label = batch['label'].to(device) # (B, seq_len)

                # Compute the loss using a simple cross entropy
                #proj_output: (B, seq_len, tgt_vocab_size) -> (B *seq_len, tgt_vocab_size)
                loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                # Log the loss
                writer.add_scalar('train loss', loss.item(), global_step)
                writer.flush()

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        
                global_step += 1

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.max_len , device, lambda msg: batch_iterator.write(msg), global_step, writer )
            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            model_folder = Path(config.model_folder)
            
            if not Path.exists(model_folder):
                os.mkdir(model_folder)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
            logger.info(f" Model saved in file path : {model_filename}")

        