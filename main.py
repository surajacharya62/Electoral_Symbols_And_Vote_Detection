from modules import logger
from modules.symbol_detection.faster_rcnn.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

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
