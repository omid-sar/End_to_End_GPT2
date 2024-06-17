from GPT2.logging import logger 
from GPT2.config.configuration import ConfigurationManager
from GPT2.components.model_inference import Inference



class ModelInferencePipeline:
    def __init__(self):

        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config()
        self.model_config = self.config_manager.get_model_config()
        self.model_training_config = self.config_manager.get_model_training_config()
        self.translator = Inference(self.data_transformation_config, self.model_config, self.model_training_config)

    def main(self, sentence: str):
        return self.translator.translate(sentence)
