from GPT2.logging import logger 
from GPT2.config.configuration import ConfigurationManager
from GPT2.components.data_transformation import DataTokenizer


class DataTransformationTrainingPipeline():
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        tokenizer = DataTokenizer(config=data_transformation_config)
        tokenizer.process_documents()

