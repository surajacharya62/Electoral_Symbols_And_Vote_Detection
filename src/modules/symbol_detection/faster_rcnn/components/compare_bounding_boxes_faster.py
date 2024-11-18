import pandas as pd
import torch
from modules.symbol_detection.faster_rcnn.utils.faster_rcnn_utils import torch_nms



class CompareBoundingBox:

    def labels(self, test_set, predictions, label_id):
        total_comparisions = []
        total_comparisions1 = []
        for i, (test_data, label) in enumerate(zip(test_set, predictions)):  
            # print(test_data, label)

            actual_labels = test_data[3]['labels']  
            true_bboxes = test_data[3]['boxes']
            image_name1 = test_data[2]

            predicted_bboxes = label[0]['boxes'] 
            predicted_labels = label[0]['labels']
            scores = label[0]['scores']
            image_name = label[2]   

            indices = torch_nms(predicted_bboxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )      

            final_boxes, final_scores, final_labels  =  predicted_bboxes[indices], scores[indices], predicted_labels[indices]   
            # print(final_boxes)
            # print(final_labels)
            final_boxes = final_boxes.tolist()
            final_labels = final_labels.tolist()
            final_scores =  final_scores.tolist()
            true_bboxes = true_bboxes.tolist()
            actual_labels = actual_labels.tolist()

            # print(final_boxes)
            # print(final_labels)

            matches1,matches2,matches3,matches4 = self.compare_labels_with_bboxes(final_labels, actual_labels, final_boxes, true_bboxes, final_scores, image_name1,label_id, iou_threshold=0.5)
            df = matches1 + matches2 + matches3 + matches4
            
            total_comparisions.append(df)
            # total_comparisions1.append(matches1)
            # total_comparisions1.append(matches2)
            # total_comparisions1.append(matches3)

        
        data = pd.DataFrame(total_comparisions)
        data.to_excel('./artifacts/faster_rcnn_files/df_total_comparisions.xlsx')
        # data1 = pd.DataFrame(total_comparisions1)
        # data1.to_excel('df_total_comparisions1.xlsx')



    def calculate_iou(self,boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Calculate the area of intersection
        intersection_area = max(0, xB - xA) * max(0, yB - yA)

        # Calculate the areas of both bounding boxes
        boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Calculate the area of union
        union_area = boxA_area + boxB_area - intersection_area

        # Compute the IoU
        iou = intersection_area / float(union_area)

        return iou


#---------------------------------
    def compare_labels_with_bboxes(self, predicted_labels, true_labels, predicted_bboxes, true_bboxes, scores, image_name,label_id, iou_threshold=0.5):
        matches1 = []
        matches2 = []
        matches3 = []  
        matches4 = [] 
        inv_label = {v:k for k,v in label_id.items()}  
        # print(true_bboxes, predicted_bboxes)
        
        for tlabel, tbox in zip(true_labels, true_bboxes):
            best_iou = 0
            value_checked = []
            
            best_match = {'true_label': None, 'iou': 0, 'valid': False}
            # print(true_labels, true_bboxes)
            # print(true_labels[0])
            # true_labels = true_labels.cpu().numpy()
            # true_bboxes = true_bboxes.cpu().numpy()
            # print(true_labels)
            # tlabel = label_id.get(tlabel, "unkown")
            tlabel = tlabel
            # print(label_id, "tlabel")

            for plabel, pbox, score in zip(predicted_labels, predicted_bboxes, scores):
               
                # if isinstance(tbox, torch.Tensor):
                #     tbox_list = tbox.tolist()  # Convert to list
                # else:
                #     tbox_list = tbox 

                # print("tbox_list:", tbox_list)
                iou = self.calculate_iou(pbox, tbox) 
                # print(plabel)
                # print(inv_label)
                # plabel = label_id.get(plabel, 'unkown')
                # print(plabel,tlabel, 'plabel', pbox,tbox)

                if iou > iou_threshold and tlabel == plabel:
                    
                    
                        # pred_label = inv_label.get(pred_label, 'unkown')
                        # if iou >= iou_threshold :
                            # best_iou = iou
                    best_match = {'image_name':image_name,
                                    'true_label':inv_label.get(tlabel,'unknown'),
                                    'pred_label':inv_label.get(plabel,'unknown'),                                 
                                    'class': inv_label.get(tlabel,'known'),
                                    'Confidence': score,
                                    'iou': iou, 
                                    'TP':  1, 
                                    'FP': 0,
                                    'FN':0}
                    matches1.append(best_match)
                
                elif iou > iou_threshold and tlabel != plabel:

                    best_match = {'image_name':image_name,
                                    'true_label':inv_label.get(tlabel,'unknown'),
                                    'pred_label':inv_label.get(plabel,'unknown'),                                 
                                    'class': inv_label.get(tlabel,'known'),
                                    'confidence': score,
                                    'iou': iou, 
                                    'TP':  0, 
                                    'FP': 1, 
                                    'FN':0} 
                    matches2.append(best_match) 
                
                
                    
               
        # pred_labels = [label_id.get(label) for label in predicted_labels]
        
        true_labels1 = true_labels
        # print(true_labels1)

        # p_list = []
        # for p_label, score in pred_labels:
        #     p_list.append(p_label)
        
        non_object_detected = set(true_labels) - set(predicted_labels)
        # print(non_object_detected)
        

        for label in non_object_detected:
            
            pred_label1 = inv_label.get(label)

            # true_label = inv_label.get(true_label,'unkown')
            # pred_label = inv_label.get(object,'unkown')
            best_match = {'image_name':image_name,
                             'true_label': inv_label.get(label, 'unkown'),
                             'pred_label':'Not detected', 
                           
                            'class':inv_label.get(label, 'unknown'),
                            'confidence': 0,                            
                                'iou': 0,
                                'TP':0, 
                                'FP': 0,
                                'FN':1}
                
            matches4.append(best_match)

        
        return matches1, matches2, matches3, matches4     
