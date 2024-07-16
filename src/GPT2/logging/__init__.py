import os
import sys
import torch
import logging
import torch.distributed as dist

def is_master_process():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

device = "cuda" if torch.cuda.is_available() else("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
absolute_path = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.abspath(os.path.join(absolute_path, "../../.."))
log_dir = os.path.join(base_dir, "logs")
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# Configure the logging
handlers = [logging.FileHandler(log_filepath)]
if device in ["cpu", "mps"] or is_master_process():
    handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=handlers
)

logger = logging.getLogger("GPT2")
