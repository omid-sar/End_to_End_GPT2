from GPT2.config.configuration import ConfigurationManager
from GPT2.components.model_training import train_model



class ModelTrainingPipeline():
    def __init__(self, train_loader, model, optimizer) -> None:
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.model = model
        self.config = ConfigurationManager()
        self.config = self.config.get_model_training_config()

    def main(self):
        train_model(
            config=self.config,
            train_loader=self.train_loader,
            model=self.model,
            optimizer=self.optimizer,
            )
    



