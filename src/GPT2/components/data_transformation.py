import os
from pathlib import Path
from datasets import load_dataset 


from GPT2.logging import logger
from GPT2.entity import DataTransformationConfig

class DataTokenizer:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        file_path = Path(os.path.join(config.local_data_file, config.dataset_name))
        if os.path.exists(file_path):
            try:
                logger.info(f"Loading dataset from cache at {file_path}")
                self.dataset = load_dataset(config.dataset, name=config.dataset_name, cache_dir=str(file_path))
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")
                return None
        else:
            logger.warning(f"No cached data found at {file_path}. Consider downloading it first.")





# -----------------------
from GPT2.config.configuration import ConfigurationManager
import os; os.chdir("../../..")
config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
tokenizer = DataTokenizer(config=data_transformation_config)
if tokenizer.check_existing_data():
    logger.info(f"Tokenized data is available. Proceeding with further processing.")
else:
    logger.info(f"No tokenized data found. Starting tokenization process.")
    tokenizer.process_documents()
    logger.info(f"Tokenization completed.")
    
tokenizer.config.shard_size