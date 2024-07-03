import os
from pathlib import Path
from datasets import load_dataset 
import tiktoken
import numpy as np


from GPT2.logging import logger
from GPT2.entity import DataTransformationConfig

class DataTokenizer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.file_path = Path(os.path.join(config.local_data_file, config.dataset_name))
        if os.path.exists(self.file_path):
            try:
                logger.info(f"Loading dataset from cache at {self.file_path}")
                self.dataset = load_dataset(config.dataset, name=config.dataset_name, cache_dir=str(self.file_path))
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
                return None
        else:
            logger.warning(f"No cached data found at {self.file_path}. Consider downloading it first.")
        
        self.enc = tiktoken.get_encoding('gpt2')
        self.eot = self.enc._special_tokens['<|endoftext|>'] # end of text token

    
    def check_existing_tokenized_data(self):
        files = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith('.npy')]
        if files:
            logger.info(f"Found {len(files)} pre-tokenized shards in {self.file_path}. Skipping tokenization.")
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










# -----------------------
from GPT2.config.configuration import ConfigurationManager
import os; os.chdir("../../..")
config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
tokenizer = DataTokenizer(config=data_transformation_config)
if tokenizer.check_existing_tokenized_data():
    logger.info(f"Tokenized data is available. Proceeding with further processing.")
else:
    logger.info(f"No tokenized data found. Starting tokenization process.")
    tokenizer.process_documents()
    logger.info(f"Tokenization completed.")
    
