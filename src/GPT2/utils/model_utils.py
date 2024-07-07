from pathlib import Path
import torch
import os
from box import ConfigBox
from GPT2.logging import logger
from torch.distributed import init_process_group

def get_device():
    device = "cuda" if torch.cuda.is_available() else("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    elif device == 'mps':
        logger.info("Device name: Apple Metal Performance Shaders (MPS)")
    else:
        logger.info("NOTE: If you have a GPU, consider using it for training.")
    return device


def get_weights_file_path(config, epoch):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}{epoch}.pt"
    weights_file_path = str(Path('.') / model_folder / model_filename)
    logger.info(f"Generated weights file path: {weights_file_path}")
    return weights_file_path



def latest_weights_file_path(config):
    model_folder = config.model_folder
    model_filename = f"{config.model_basename}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if not weights_files:
        logger.info(f"No weights files found in {model_folder}. Starting from scratch")
        return None
    weights_files.sort()
    latest_file = str(weights_files[-1])
    logger.info(f"Latest weights file found: {latest_file}")
    return latest_file


def save_model_summary(model, file_path, input_size, device='cpu'):
    """
    Saves the model summary to a file.
    """
    original_device = next(model.parameters()).device
    model.to(device)

    try:
        with open(file_path, 'w') as f:
            # Here you would generate the model summary.
            # For now, we're just simulating this by writing a placeholder string.
            f.write("Model summary placeholder")
        logger.info(f"Model summary saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save model summary: {e}")
    finally:
        model.to(original_device)

def save_initial_weights(model, file_path):
    """
    Saves the initial weights of the model.
    """
    try:
        torch.save(model.state_dict(), file_path)
        logger.info(f"Model initial weights saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save initial weights: {e}")


# Simple launch: =>>> 
# python main.py
# DDP launch for e.g.b 8 GPUs: =>>> 
# torchrun --standalone --nproc_per_node=8 main.py


def setup_distributed():
    # Set up DDP (Distributed Data Parallel)
    # torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        assert torch.cuda.is_available(), "for now we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # Non-DDP run, Vanilla
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    # Log device information
    logger.info(f"Using device: {device}")
    if device.startswith('cuda'):
        logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    elif device == 'mps':
        logger.info("Device name: Apple Metal Performance Shaders (MPS)")
    else:
        logger.info("NOTE: If you have a GPU, consider using it for training.")

    # Determine device_type
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    return ConfigBox({
        'ddp': ddp,
        'ddp_rank': ddp_rank,
        'ddp_local_rank': ddp_local_rank,
        'ddp_world_size': ddp_world_size,
        'master_process': master_process,
        'device': device,
        'device_type': device_type
    })