from modules.symbol_detection.faster_rcnn.components.electoral_symbol_dataset import  ElectoralSymbolDataset
from modules.symbol_detection.faster_rcnn.components.visualize_symbols_detection import VisualizePrediction
from modules.symbol_detection.faster_rcnn.components.compare_bounding_boxes_faster import CompareBoundingBox
from modules.symbol_detection.faster_rcnn.components.reshape_data import ReshapeData
from modules.symbol_detection.faster_rcnn.components.metrics import Metrics
from modules.vote_validation.faster_rcnn.validate_vote import ValidateVote
from modules.symbol_detection.faster_rcnn.utils.faster_rcnn_utils import label_to_id,get_transform,collate_fn
from torch.utils.data import DataLoader
from modules.symbol_detection.faster_rcnn.entity.config_entity import EvaluationConfig
import torch
import os


class Evalaution:  

    def __init__(self, config:EvaluationConfig):
        self.config = config
        self.true_annotation_labels = label_to_id(self.config.annotations_path)

    def get_data_loader(self):
        annotation_labels = label_to_id(self.config.annotations_path)
        test_set = ElectoralSymbolDataset(        
            self.config.test_images_path,
            "single_image",
            self.config.annotations_path,
            annotation_labels,
            get_transform(train=False),
            is_single_image=False 
        )
        
        test_data_loader = DataLoader(test_set, batch_size=self.config.params_batch_size, shuffle=False, collate_fn=collate_fn)

        return test_set, test_data_loader
    
    
    # def get_faster_rcnn_model(self):
    #     """
    #     Initializing model  
    #     """
    #     model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)    
    #     # get number of input features for the classifier
    #     in_features = model.roi_heads.box_predictor.cls_score.in_features
    #     # replace the pre-trained head with a new one
    #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.classes) 
    #     return model
    
    
    def make_predictions(self,dataset_loader, faster_rcnn_model):
        """
        Make predictions on test images
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
        faster_rcnn_model.load_state_dict(torch.load(self.config.path_of_model, map_location=device))
        faster_rcnn_model.eval()
        # test_data_loader = self.get_data_loader()
       
        predictions = []
        with torch.no_grad():
            for images, imageids, imagenames, target in dataset_loader:
                images = [image.to(device) for image in images]
                outputs = faster_rcnn_model(images)

                for output, imageid, imagename in zip(outputs, imageids, imagenames):
                    prediction = (output, imageid, imagename)
                    predictions.append(prediction)
        
        test_images_path = self.config.test_images_path
        test_images_name = []
        for filename in os.listdir(test_images_path):
            test_images_name.append(filename) 
        
        return predictions, test_images_name
    
    
    def visualize_predictions(self, predictions, test_set):
        """
            Visualize prediction
        """ 
            
        visualize = VisualizePrediction()        
        visualize.visualize_predicted_images(self.config.test_images_path, test_set, predictions, self.true_annotation_labels)
    
    
    def vote_validation(self, test_set, test_images, predictions):
        
         #-------------Vote Validation

        vote_validate = ValidateVote()        
        vote_validate.validate_vote(test_set, test_images, predictions, self.true_annotation_labels, self.config.test_images_path)

    
    def metrics_calculation(self, test_set, predictions):
    
        #Predictions Bounding Box Comparison        
        compare_bboxes = CompareBoundingBox()
       
        compare_bboxes.labels(test_set, predictions, self.true_annotation_labels) 

        #Data Reshaping
        reshape_data = ReshapeData()
        reshape_data.process_and_reshape_data_v2(self.config.faster_rcnn_files_path)

        metrics = Metrics()
        metrics.metrics(predictions, self.config.annotations_path, self.true_annotation_labels, self.config.faster_rcnn_files_path)
        metrics.call_metrics(self.config.faster_rcnn_files_path)



    


    

    