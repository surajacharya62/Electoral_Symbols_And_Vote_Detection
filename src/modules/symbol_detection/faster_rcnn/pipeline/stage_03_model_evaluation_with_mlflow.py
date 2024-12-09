from modules.symbol_detection.faster_rcnn.config.configuration import ConfigurationManager
from modules.symbol_detection.faster_rcnn.components.model_evaluation import Evalaution
from modules.symbol_detection.faster_rcnn.components.prepare_base_model import PrepareBaseModel
from modules import logger


STAGE_NAME = "Model evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config() 

        evaluate = Evalaution(config=evaluation_config)
        test_set, dataset_loader = evaluate.get_data_loader() 

        base_model_config = config.get_prepare_base_model_config()   
        base_model = PrepareBaseModel(config=base_model_config)
        model = base_model.get_prepare_faster_rcnn_model()
        
        model_predictions, images_name=evaluate.make_predictions(dataset_loader, model)

        evaluate.visualize_predictions(model_predictions, test_set)

        evaluate.metrics_calculation(test_set,model_predictions)

        # evaluate.log_into_mlflow()

        # evaluate.vote_validation(test_set, images_name, model_predictions)
        

if __name__ == "__main__":
    try:
        logger.info(f"**************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(">>>>>> stage {STAGE_NAME} completed <<<<\n\nx======x")
    except Exception as e:
        logger.exception(e)
        raise e

