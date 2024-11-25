
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from modules.symbol_detection.faster_rcnn.entity.config_entity import PrepareBaseModelConfig


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        

    def get_prepare_faster_rcnn_model(self):
        """
        Initializing model  
        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)    
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.params_classes) 
        return model