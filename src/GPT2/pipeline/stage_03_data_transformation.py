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

        if tokenizer.check_existing_data():
            # 
            # ADDING THE NUMBER OF SHARDED FILE ALREADY EXIST
            # 
            logger.info(f"Tokenized data is available. Proceeding with further processing.")
        else:
            logger.info(f"No tokenized data found. Starting tokenization process.")
            tokenizer.process_documents()
            # 
            # ADDING THE NUMBER OF SHARDED FILE ALREADY EXIST
            # 
            logger.info(f"Tokenization completed.")
            

