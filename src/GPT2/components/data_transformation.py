import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import torch

from GPT2.logging import logger
from GPT2.entity import DataTransformationConfig
from GPT2.utils.common import  get_directory_size
import tiktoken

class DataTokenizer:
    def __init__(self, config:DataTransformationConfig ):
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
            logger.info(f"Found {len(files)} pre-tokenized shards with total size {get_directory_size(self.transformed_file_path)} in {self.transformed_file_path} >>>> Skipping tokenization.")
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

    def process_single_document(self, doc):
        return self.tokenize(doc)

    def process_documents_sequential(self):
        if self.check_existing_tokenized_data():
            return

        os.makedirs(self.transformed_file_path, exist_ok=True)
        logger.info(f"Created directory at: {self.transformed_file_path}")

        shard_index = 0
        all_tokens_np = np.empty((self.config.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for doc in tqdm(self.dataset, desc="Processing documents"):
            tokens = self.process_single_document(doc)
            if token_count + len(tokens) < self.config.shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
                remainder = self.config.shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                self.write_datafile(filename, all_tokens_np)
                shard_index += 1
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(self.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
            self.write_datafile(filename, all_tokens_np[:token_count])

def process_documents_parallel(tokenizer):
    if tokenizer.check_existing_tokenized_data():
        return

    os.makedirs(tokenizer.transformed_file_path, exist_ok=True)
    logger.info(f"Created directory at: {tokenizer.transformed_file_path}")

    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((tokenizer.config.shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenizer.tokenize, tokenizer.dataset, chunksize=16):
                if token_count + len(tokens) < tokenizer.config.shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=tokenizer.config.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(tokenizer.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
                    remainder = tokenizer.config.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    tokenizer.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(tokenizer.transformed_file_path, f"edufineweb_{split}_{shard_index:06d}")
            tokenizer.write_datafile(filename, all_tokens_np[:token_count])


class DataLoaderLite:
    def __init__(self, config):
        self.B = config.B
        self.T = config.T

        with open ("tiny_shakespeare.txt", "r") as file:
            text = file.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        logger.info(f"loaded {len(self.tokens)} tokens")
        logger.info(f"loaded {len(self.tokens) // (self.B*self.T)} batches")

        self.current_position = 0

    def next_batch(self):
        buf = self.tokens[self.current_position : self.current_position+self.B*self.T+1 ]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_position += self.B * self.T
        if self.current_position > len(self.tokens):
            self.current_position = 0 
        return x, y

