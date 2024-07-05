from GPT2.logging import logger 
from GPT2.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from GPT2.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from GPT2.pipeline.stage_03_model_verification import ModelVerificationTrainingPipeline
from GPT2.pipeline.stage_04_model_training import ModelTrainingPipeline
#from GPT2.pipeline.stage_06_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    train_loader = data_transformation.main(use_multiprocessing=False) #*** should be True for GPUs
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Validation stage"
try:
    logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
    model_verification_pipeline = ModelVerificationTrainingPipeline()
    model, optimizer = model_verification_pipeline.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Model Training stage"
try:
    logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training = ModelTrainingPipeline(train_loader, model, optimizer)
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 
import sys; sys.exit(0)
STAGE_NAME = "Model Evaluation stage"
try:
    logger.info(f"\n\nx{'=' * 80}x \n\n>>>>>> stage {STAGE_NAME} started <<<<<<")
    model_training = ModelEvaluationPipeline()
    model_training.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx{'=' * 80}x")
except Exception as e:
    logger.exception(e)
    raise e 
