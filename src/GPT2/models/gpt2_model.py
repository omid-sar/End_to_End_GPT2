from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# ----------- Temporary to load Config localy here not main.py -----
from GPT2.config.configuration import ConfigurationManager
import os
os.getcwd()
os.chdir('../../../')
config = ConfigurationManager()
# -------------------------------------------------------------------

@dataclass 
class GPTconfig:
    block_size : 
