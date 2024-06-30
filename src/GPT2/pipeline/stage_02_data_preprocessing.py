from GPT2.logging import logger 
from GPT2.config.configuration import ConfigurationManager
from GPT2.components.data_preprocessing import DataPreprocessing


class DataPreprocessingTrainingPipeline():
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)
