import torch
from GPT2.utils.model_utils import get_device
from GPT2.config.configuration import ConfigurationManager
from GPT2.logging import logger
from GPT2.models.gpt2_model import GPT


class ModelVerificationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        try:
            config_manager = ConfigurationManager()
            
            config = config_manager.get_gpt_config()
            device = get_device()
            model = GPT(config=config)
            model.to(device)
            optimizer = model.configure_optimizer(weight_decay=config.weight_decay, learning_rate=config.learning_rate, device_type=device)

            # Optionally, perform a simple forward pass check
            dummy_config = config_manager.get_data_transformation_config()
            dummy_input = torch.randint(0, config.vocab_size, (dummy_config.B, dummy_config.T)).to(device)
            with torch.no_grad():
                logits, _ = model(dummy_input)
            assert logits.size() == (dummy_config.B, dummy_config.T, config.vocab_size) , f"Model size failed"
            logger.info("Basic forward pass successful.")

            return model, optimizer

        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            raise e

