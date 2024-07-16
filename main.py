import sys
import torch
from GPT2.logging import logger
from GPT2.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from GPT2.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from GPT2.pipeline.stage_03_model_training import ModelTrainingPipeline
#from GPT2.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from GPT2.utils.model_utils import setup_distributed

def main():
    dist_config = setup_distributed() 

    if dist_config.master_process:
        STAGE_NAME = "Data Ingestion stage"
        logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")

        STAGE_NAME = "Data Transformation stage"
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main(use_multiprocessing=True)  #***Change as needed
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")

    if dist_config.ddp:
        torch.distributed.barrier()

    STAGE_NAME = "Model Training stage"
    try:
        if dist_config.master_process:
            logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        model_training = ModelTrainingPipeline(use_compile=False)
        model_training.main()
        if dist_config.master_process:
            logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
    except Exception as e:
        if dist_config.master_process:
            logger.exception(e)
        raise e 
    # STAGE_NAME = "Model Evaluation stage"
    # try:
    #     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    #     model_training = ModelEvaluationPipeline()
    #     model_training.main()
    #     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
    # except Exception as e:
    #     logger.exception(e)
    #     raise e 

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("An error occurred in the main execution:")
        sys.exit(1)