import torch
from GPT2.utils.model_utils import  setup_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
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
            model = GPT(config=config)
            dist_config = setup_distributed()
            device = dist_config.device
            model.to(device)
            #***model = torch.compile(model) # torch.compile interferes with HellaSwag eval and Generation. there is problem here
            if dist_config.ddp:
                 model = DDP(model, device_ids=[dist_config.ddp_local_rank])
            # a consistent way to access the underlying model, whether it's wrapped in DDP or not.
            raw_model = model.module if dist_config.ddp else model 
            optimizer = raw_model.configure_optimizer(weight_decay=config.weight_decay, learning_rate=config.learning_rate, betas=config.betas ,device_type=device)

            # Optionally, perform a simple forward pass check
            dummy_config = config_manager.get_data_transformation_config()
            dummy_input = torch.randint(0, config.vocab_size, (dummy_config.B, dummy_config.T)).to(device)
            with torch.no_grad():
                logits, _ = model(dummy_input)
            assert logits.size() == (dummy_config.B, dummy_config.T, config.vocab_size) , f"Model size failed"
            logger.info("Basic forward pass successful.")

            return model, optimizer, raw_model

        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            raise e

