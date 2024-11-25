from modules import logger
from modules.symbol_detection.faster_rcnn.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from modules.symbol_detection.faster_rcnn.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from modules.symbol_detection.faster_rcnn.pipeline.stage_03_model_evaluation import EvaluationPipeline

# logger.info("welcome to electorals and sysmbols detection project")

STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<<\nx==================x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare base model stage"

if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<\n\nx===================x")
    except Exception as e:
        logger.exception(e)
        raise e
    

STAGE_NAME = "Model evaluation stage"

if __name__ == "__main__":
    try:
        logger.info(f"**************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e