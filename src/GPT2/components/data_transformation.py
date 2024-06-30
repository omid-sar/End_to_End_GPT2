import os
from pathlib import Path
from datasets import load_dataset 


from GPT2.logging import logger
from GPT2.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
    


# -----------------------
"""from GPT2.config.configuration import ConfigurationManager
import os; os.chdir("../../..")
config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
data_transformation = DataTransformation(config=data_transformation_config)
"""