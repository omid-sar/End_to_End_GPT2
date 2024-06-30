import os
from pathlib import Path
from datasets import load_dataset 


from GPT2.logging import logger
from GPT2.entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
    


# -----------------------
"""from GPT2.config.configuration import ConfigurationManager
import os; os.chdir("../../..")
config = ConfigurationManager()
data_preprocessing_config = config.get_data_preprocessing_config()
data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
"""