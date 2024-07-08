from GPT2.config.configuration import ConfigurationManager
from GPT2.components.model_training import train_model
from GPT2.utils.model_utils import setup_distributed



class ModelTrainingPipeline():
    def __init__(self, train_loader, model, optimizer, raw_model) -> None:
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.raw_model = raw_model
        self.model = model
        self.config = ConfigurationManager()
        self.config = self.config.get_model_training_config()
        self.dist_config = setup_distributed()

    def main(self):
        train_model(
            config=self.config,
            train_loader=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            raw_model = self.raw_model,
            dist_config = self.dist_config
            )
    



