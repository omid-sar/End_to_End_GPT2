from GPT2.constants import *
from GPT2.utils.common import read_yaml, create_directories
from GPT2.entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig
from GPT2.entity import GPTConfig, ModelTrainingConfig, ModelEvaluationConfig, ModelInferenceConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH):

        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            dataset = config.dataset,
            local_data_file = config.local_data_file
        )

        return data_ingestion_config
    

    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            dataset_name = config.dataset_name,
            dataset = config.dataset,
            downloaded_files  = config.downloaded_files, 
            local_data_file = config.local_data_file,
            shard_size = config.shard_size,
            total_batch_size = config.total_batch_size,
            B = config.B,
            T = config.T,
        )

        return data_transformation_config


    def get_gpt_config(self) -> GPTConfig:
        config = self.config.gpt_config

        create_directories([config.root_dir])
        create_directories([config.verification_info_dir])
        

        return GPTConfig(
            root_dir = config.root_dir,
            verification_info_dir = config.verification_info_dir,
            verification_summary_file = config.verification_summary_file, 
            verification_weights_file = config.verification_weights_file, 
            block_size = config.block_size,
            vocab_size = config.vocab_size,
            n_layer = config.n_layer,
            n_head = config.n_head,
            n_embd = config.n_embd,
            weight_decay = config.weight_decay,
            learning_rate = config.learning_rate,
            betas = config.betas
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training

        create_directories([config.root_dir])

        model_training_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            model_folder = config.model_folder,
            total_batch_size = config.total_batch_size,
            B = config.B,
            T = config.T,
            max_lr = config.max_lr,
            min_lr = config.min_lr,
            warmup_steps = config.warmup_steps,
            max_steps = config.max_steps,
        )

        return model_training_config
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir = config.root_dir,
    
            
        )

        return model_evaluation_config
    

    def get_model_inference_config(self) -> ModelInferenceConfig:
        config = self.config.model_inference

        create_directories([config.root_dir])

        model_inference_config = ModelInferenceConfig(
            root_dir = config.root_dir,
    
            
        )

        return model_inference_config
    

