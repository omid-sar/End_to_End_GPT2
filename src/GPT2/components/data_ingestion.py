import os
from pathlib import Path
from datasets import load_dataset 


from GPT2.logging import logger
from GPT2.utils.common import get_size, create_directories
from GPT2.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        file_path = Path(os.path.join(self.config.local_data_file, self.config.dataset_name))
        if not file_path.exists():
            os.makedirs(self.config.local_data_file, exist_ok=True)
            data = load_dataset(self.config.dataset, name=self.config.dataset_name, split="train", cache_dir=str(file_path))
            logger.info(f"{self.config.dataset_name} downloaded!")
        else:
            logger.info(f"Dataset already exists at {self.config.local_data_file}. Size: {get_size(file_path)}")

