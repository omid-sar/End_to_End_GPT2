from GPT2.logging import logger 
from GPT2.config.configuration import ConfigurationManager
from GPT2.components.data_transformation import DataTokenizer, process_documents_parallel, DataLoaderLite
from GPT2.utils.model_utils import setup_distributed


class DataTransformationTrainingPipeline():
    def __init__(self) -> None:
        pass

    def main(self, use_multiprocessing=False):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            tokenizer = DataTokenizer(config=data_transformation_config)
            dist_config = setup_distributed()
            num_processes = dist_config.ddp_world_size
            process_rank = dist_config.ddp_rank
            train_loader = DataLoaderLite(config=data_transformation_config,
                                          process_rank = dist_config.ddp_rank,
                                          num_processes = dist_config.ddp_world_size
                                          )
        
            logger.info(f"Starting data transformation with multiprocessing={'enabled' if use_multiprocessing else 'disabled'}")
            
            if use_multiprocessing:
                process_documents_parallel(tokenizer)
            else:
                tokenizer.process_documents_sequential()
            return train_loader
        
        except Exception as e:
            logger.error(f"An error occurred during data transformation: {str(e)}")
            raise
        