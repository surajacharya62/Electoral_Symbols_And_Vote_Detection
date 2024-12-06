import torch
import os
from torch.utils.data import DataLoader

from modules.symbol_detection.faster_rcnn.components.electoral_symbol_dataset import  ElectoralSymbolDataset
from modules.symbol_detection.faster_rcnn.components.prepare_base_model import PrepareBaseModel
from modules.symbol_detection.faster_rcnn.config.configuration import ConfigurationManager
from modules.symbol_detection.faster_rcnn.components.visualize_symbols_detection import VisualizePrediction
from modules.symbol_detection.faster_rcnn.entity.config_entity import EvaluationConfig
from modules.vote_validation.faster_rcnn.validate_vote import ValidateVote
from modules.symbol_detection.faster_rcnn.utils.faster_rcnn_utils import label_to_id,get_transform,collate_fn




class PredictionPipeline:
    def __init__(self):
        self.predictions = ""
        self.config = ConfigurationManager()
        self.evaluation_config = self.config.get_evaluation_config() 
        self.base_model_config = self.config.get_prepare_base_model_config() 
        self.true_annotation_labels = label_to_id(self.evaluation_config.annotations_path)          
        

      

    def predict(self, image_file):
        """
        Make predictions on test images
        """
        
        base_model = PrepareBaseModel(config=self.base_model_config)
        self.model = base_model.get_prepare_faster_rcnn_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
        self.model.load_state_dict(torch.load(self.base_model_config.base_model_path, map_location=device))
        self.model.eval()
        
        # predictions = []
        with torch.no_grad():
            self.predictions = self.model(image_file)
            # self.predictions.append(predictions) 
            # print(self.predictions[0]['boxes'])
            # print(self.predictions[0]['labels'])
            # print(self.predictions[0])
        

    def visualize(self, img, image_name):    
        visualize = VisualizePrediction()        
        visualize.visualize_single_image(img,image_name, self.predictions, self.true_annotation_labels)
    

    def validate_vote(self, img, image_name):    
        visualize = ValidateVote()        
        visualize.validate_single_ballot(img,image_name, self.predictions, self.true_annotation_labels,self.evaluation_config.annotations_path)
       

    
  