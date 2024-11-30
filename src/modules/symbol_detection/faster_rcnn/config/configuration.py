from modules.symbol_detection.faster_rcnn.constants import *
from src.utils.common import read_yaml,create_directories
from modules.symbol_detection.faster_rcnn.entity.config_entity import (DataIngestionConfig,PrepareBaseModelConfig,EvaluationConfig)


class ConfigurationManager:
    def __init__(
            self,
            config_filepath =  CONFIG_FILE_PATH,
            params_filepath = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
            
            )        
        return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            # updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
    
    
    def get_evaluation_config(self)->EvaluationConfig:
      
      config = self.config.base_model

      eval_config = EvaluationConfig(
                root_dir = Path(config.root_dir),
                path_of_model=Path(config.base_model_path),
                test_images_path=Path(config.test_images_path),
                annotations_path=Path(config.annotations_path),
                faster_rcnn_files_path=Path(config.faster_rcnn_files_path),
                mlflow_uri="https://dagshub.com/surajacharya62/Electoral_Symbols_And_Vote_Detection.mlflow",
                all_params=self.params,
                params_image_size=self.params.IMAGE_SIZE,
                params_batch_size=self.params.BATCH_SIZE,
                # classes=self.params.CLASSES                
          )
      
      return eval_config
