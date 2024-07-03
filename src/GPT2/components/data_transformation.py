import os
from pathlib import Path
from datasets import load_dataset 
import multiprocessing as mp
import tiktoken, tqdm
import numpy as np


from GPT2.logging import logger
from GPT2.entity import DataTransformationConfig
from GPT2.utils.common import create_directories

class DataTokenizer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.transformed_file_path = Path(os.path.join(self.config.local_data_file, self.config.dataset_name))
        file_path = Path(os.path.join(config.downloaded_files, config.dataset_name))
    
        if os.path.exists(file_path):
            try:
                logger.info(f"Loading dataset from cache at {file_path}")
                self.dataset = load_dataset(config.dataset, name=config.dataset_name, cache_dir=str(file_path))
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
                return None
        else:
            logger.warning(f"No cached data found at {file_path}. Consider downloading it first.")
        
        self.enc = tiktoken.get_encoding('gpt2')
        self.eot = self.enc._special_tokens['<|endoftext|>'] # end of text token

    
    def check_existing_tokenized_data(self):
        if not os.path.exists(self.transformed_file_path):
            logger.info(f"No tokenized data directory found at {self.transformed_file_path}. Starting tokenization.")
            return False
        files = [os.path.join(self.transformed_file_path, f) for f in os.listdir(self.transformed_file_path) if f.endswith('.npy')]
        if files:
            logger.info(f"Found {len(files)} pre-tokenized shards in {self.transformed_file_path}. Skipping tokenization.")
            return True
        else:

            return False
        
    def check_existing_tokenized_data(self):
        if not os.path.exists(self.transformed_file_path):
            logger.info(f"No tokenized data directory found at {self.transformed_file_path}. Starting tokenization.")
            return False
        files = [os.path.join(self.transformed_file_path, f) for f in os.listdir(self.transformed_file_path) if f.endswith('.npy')]
        if files:
            logger.info(f"Found {len(files)} pre-tokenized shards in {self.transformed_file_path}. Skipping tokenization.")
            return True
        else:
            return False
        
    def tokenize(self, doc):
        # Tokenizes a single document and returns a numpy array of UNIT16 tokens
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0  <= tokens_np).all() and (tokens_np <= 2**16).all(), "Token dictionary too large for unit16"
        tokens_np_unit16 = tokens_np.astype(np.uint16)
        return tokens_np_unit16
    
    def write_datafile(self, file_name, tokens_np):
        np.save(file_name, tokens_np)

    def process_documents(self):
        if self.check_existing_tokenized_data():
            return # Skip documents tokenizer if already exist!
        
        os.makedirs(self.transformed_file_path, exist_ok=True)
        logger.info(f"created directory at: {self.transformed_file_path}")
    
        nprocs = max(1, os.cpu_count()//2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty(shape=(self.config.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            
            for tokens in pool.imap(self.tokenize, self.dataset, chunksize=16):
                if token_count + len(tokens) < self.config.shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar == None:
                        progress_bar = tqdm(total = self.config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")



# -----------------------
from GPT2.config.configuration import ConfigurationManager
import os; os.chdir("../../..")
config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
tokenizer = DataTokenizer(config=data_transformation_config)
tokenizer.process_documents()