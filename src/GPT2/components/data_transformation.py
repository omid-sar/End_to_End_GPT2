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
        np.save(filename, tokens_np, allow_pickle=False)

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
                filename = os.path.join(self.transformed_file_path, f"{self.config.dataset}_{split}_{shard_index:06d}")
                remainder = self.config.shard_size - token_count
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                self.write_datafile(filename, all_tokens_np)
                shard_index += 1
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(self.transformed_file_path, f"{self.config.dataset}_{split}_{shard_index:06d}")
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
                    filename = os.path.join(tokenizer.transformed_file_path, f"{tokenizer.config.dataset}_{split}_{shard_index:06d}")
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
            filename = os.path.join(tokenizer.transformed_file_path, f"{tokenizer.config.dataset}_{split}_{shard_index:06d}")
            tokenizer.write_datafile(filename, all_tokens_np[:token_count])

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

