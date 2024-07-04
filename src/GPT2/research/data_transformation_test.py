import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

from GPT2.logging import logger


class Config:
    def __init__(self):
        self.root_dir = 'artifacts/data_transformation'
        self.dataset_name = 'wikitext-2-raw-v1'
        self.dataset = 'wikitext'
        self.downloaded_files = 'artifacts/data_ingestion/data'
        self.local_data_file = 'artifacts/data_transformation/data'
        self.shard_size = 1000000  # 1M tokens per shard

class DataTokenizer:
    def __init__(self, config):
        self.config = config
        self.transformed_file_path = Path(os.path.join(self.config.local_data_file, self.config.dataset_name))
        file_path = Path(os.path.join(config.downloaded_files, config.dataset_name))
        
        try:
            logger.info(f"Loading dataset from cache at {file_path}")
            self.dataset = load_dataset(config.dataset, name=config.dataset_name, split="train", cache_dir=str(file_path))
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None
        
        self.enc = tiktoken.get_encoding('gpt2')
        self.eot = self.enc._special_tokens['<|endoftext|>']  # end of text token

    def check_existing_tokenized_data(self):
        if not os.path.exists(self.transformed_file_path):
            logger.info(f"No tokenized data directory found at {self.transformed_file_path}. Starting tokenization.")
            return False
        files = [os.path.join(self.transformed_file_path, f) for f in os.listdir(self.transformed_file_path) if f.endswith('.npy')]
        if files:
            logger.info(f"Found {len(files)} pre-tokenized shards in {self.transformed_file_path}. Skipping tokenization.")
            return True
        return False

    def tokenize(self, doc):
        tokens = [self.eot]  # the special <|endoftext|> token delimits all documents
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def process_documents(self):
        if self.check_existing_tokenized_data():
            return  # Skip documents tokenizer if already exist!
        
        os.makedirs(self.transformed_file_path, exist_ok=True)
        logger.info(f"Created directory at: {self.transformed_file_path}")
    
        nprocs = max(1, os.cpu_count()//2)
        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.config.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            
            for tokens in pool.imap(self.tokenize, self.dataset, chunksize=16):
                if token_count + len(tokens) < self.config.shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=self.config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
                    remainder = self.config.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder

            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
                self.write_datafile(filename, all_tokens_np[:token_count])

def main():
    config = Config()
    tokenizer = DataTokenizer(config=config)
    tokenizer.process_documents()

#if __name__ == '__main__':
    #mp.freeze_support()
main()