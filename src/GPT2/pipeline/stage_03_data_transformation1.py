
import os
import torch
from pathlib import Path

from GPT2.logging import logger
from GPT2.config.configuration import ConfigurationManager
from GPT2.components.data_transformation1 import get_ds


class DataTransformationTrainingPipeline():
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_tranformation_config = config.get_data_transformation_config()
        train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config=data_tranformation_config)

        save_dir = Path(data_tranformation_config.root_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(train_dataloader, save_dir / 'train_dataloader.pth')
        torch.save(val_dataloader, save_dir / 'val_dataloader.pth')

        logger.info("Data transformation stage completed and outputs saved.")
        return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
