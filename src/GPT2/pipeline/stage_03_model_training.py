from GPT2.config.configuration import ConfigurationManager
from GPT2.components.model_training import train_model, DataLoaderLite
from GPT2.utils.model_utils import setup_distributed



class ModelTrainingPipeline():
    def __init__(self,  use_compile) -> None:
        self.use_compile = use_compile
        self.config = ConfigurationManager()
        self.training_config = self.config.get_model_training_config()
        self.gpt_config = self.config.get_gpt_config()
        self.data_transformation_config = self.config.get_data_transformation_config()
        self.dist_config = setup_distributed()

    def main(self):
        train_model(
            training_config = self.training_config,
            gpt_config = self.gpt_config,
            data_transformation_config = self.data_transformation_config,
            dist_config = self.dist_config, 
            use_compile = self.use_compile,
            )
    