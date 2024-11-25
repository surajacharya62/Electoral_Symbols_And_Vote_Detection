
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import math
import warnings
from collections import Counter
import os
from PIL import Image
from modules.symbol_detection.faster_rcnn.utils.faster_rcnn_utils import torch_nms, reconstruct_grid_cells,is_stamp_valid, calculate_iou, is_stamp_for_symbol

warnings.filterwarnings('ignore', '.*clipping input data.*')




class ValidateVote():

    def __init__(self):
        pass  
    
    def validate_vote(self,test_set, test_images, pred_labels, label_to_id, test_images_path):   
        """
        validating the vote whether it is valid or invalid comparing the distance between stamp bounding box with each symbols boundin box
        """
        # args =  parse_args()
        # config = load_config(args.config)
        margins = (1560, 300, 200, 200)  # top, bottom, left, right margins
        ballot_size = (2668, 3413)  # width, height of the ballot paper
        symbol_size = (189, 189)  # width, height of the symbols
        rows = 7  # Number of symbol rows
        columns = 6  # Number of symbol columns  
        candidates = 42
        grid_cells = reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns)  
        id_to_label = {value: key for key, value in label_to_id.items()}
        valid_stamp_id = 40
        invalid_stamp_id = 15
        results = []

        for i, (test_data, test_image, prediction) in enumerate(zip(test_set, test_images, pred_labels)):   
            # Convert tensor image to numpy array
            # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            # img = test_data[0] 
            image = os.path.join(test_images_path, test_image)            
            image = Image.open(image)    
            image_name = test_data[2]  
            # print(image_name)        
            # img_np = img.permute(1, 2, 0).numpy() 
            # img_np = np.clip(img_np, 0, 1)  #Ensure the image array is between 0 and 1
            
            fig, ax = plt.subplots(1)
            ax.imshow(image) 
            # print(img_np)

            boxes = prediction[0]['boxes'] 
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            image_name1 = prediction[2]
            # print(image_name, image_name1)            

            actual_labels = test_data[3]['labels'].cpu().numpy()
            actual_bboxes = test_data[3]['boxes'].cpu().numpy()
            true_labels = zip(actual_labels,actual_bboxes)            

            indices = torch_nms(boxes, scores)   
            final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices] 
            # final_boxes, final_scores, final_labels = visualize.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )  
                     
            actual_label = 0
            
            if valid_stamp_id in actual_labels:
                actual_label = 'valid'
                actual_label_id = 40
            else:
                actual_label = 'invalid'   
                actual_label_id = 15                                 
            
            for box, score, pred_label in zip(final_boxes, final_scores, final_labels):
                               
                # if score2 > 0.5:
                bounding_box = box.cpu().numpy()
                # bounding_box = [round(coordinate) for coordinate in bounding_box]
                # print('box1')
                # print(label)
                label_id = int(pred_label.cpu().numpy())

                label_ids = [label for label in final_labels]
                counts = Counter(label_ids)

                both_valid_invalid_check = [15, 40]       
                total_count_valid_invalid = sum(counts[num] for num in both_valid_invalid_check)

                count_15 = label_ids.count(15)
                count_40 = label_ids.count(40)

                if label_id == valid_stamp_id:    
                    # print("stamp")                    
                    if is_stamp_valid(bounding_box, grid_cells):
                        # print("valid_stamp")
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        label_idx = int(pred_label.cpu().numpy())
                        class_name = id_to_label.get(label_idx, 'Unknown') 
                        ax.text(x1, y1, class_name, color='blue', fontsize=8)
                        
                        is_valid_symbol, symbol_label, symbol_box, closet_distance = is_stamp_for_symbol(bounding_box, zip(final_boxes, final_labels, final_scores), image_name)
                        # print(is_valid_symbol)
                        if is_valid_symbol:
                            # print("vaid symbol")
                            x1, y1, x2, y2 = symbol_box
                            label_idy = int(symbol_label)
                            class_name = id_to_label.get(label_idy, 'Unknown')
                            filtered_labels_and_bboxes = [(label, bbox) for label, bbox in zip(actual_labels, actual_bboxes) if label == symbol_label]
                            # print(filtered_labels_and_bboxes, 'filterboxes')
                            # true_label, true_box = filtered_labels_and_bboxes[0],filtered_labels_and_bboxes[1]
                            bounding_boxes = [bbox for _, bbox in filtered_labels_and_bboxes]
                            # t_box = []
                            # Printing the bounding boxes
                            for bbox in bounding_boxes:
                                t_box = bbox
                                break
                            t_box = [int(coordinate) for coordinate in t_box]

                            stamp_box = [(label, bbox) for label, bbox in zip(actual_labels, actual_bboxes) if label == label_id]
                            # print(filtered_labels_and_bboxes, 'filterboxes')
                            # true_label, true_box = filtered_labels_and_bboxes[0],filtered_labels_and_bboxes[1]
                            bounding_boxes_stamp = [bbox for _, bbox in stamp_box]
                            # t_box = []
                            # Printing the bounding boxes
                            for bbox in bounding_boxes_stamp:
                                t_box_stamp = bbox
                                break
                            t_box_stamp = [int(coordinate) for coordinate in t_box_stamp]

                            if calculate_iou(symbol_box, t_box) > 0.5 and label_id == valid_stamp_id:
                                print(image_name, actual_labels, label_id, actual_label_id, '----------------------------------' )
                                
                                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                                ax.add_patch(rect)
                                # Add label text
                                # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                                ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
                                
                                plt.axis('off')  # Optional: Remove axes for cleaner visualization
                                plt.savefig(f'./output/vote_validation/faster_rcnn/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0, dpi=600)
                                plt.close()
                                results.append((image_name, class_name, actual_label, 'valid', 'valid','valid stamp and valid symbol', closet_distance))
                                print('Symbols' + str(id_to_label))
                                print('Vote: ' + class_name + "|" + "valid|" + "valid stamp and valid symbol")
                                # plt.close()
                            else:
                                
                                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                                ax.add_patch(rect)
                                # Add label text
                                # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                                ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
                                
                                plt.axis('off')  # Optional: Remove axes for cleaner visualization
                                plt.savefig(f'./output/vote_validation/faster_rcnn/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                                plt.close()
                                results.append((image_name, class_name, actual_label,'invalid', 'invalid', 'invalid symbol or invalid stamp', closet_distance))
                                print(image_name, actual_labels, label_id, stamp_box )
                                print('Symbols' + str(id_to_label))
                                print('Vote: ' + class_name + "|" + "invalid|" + "invalid symbol or invalid stamp")
                      
                        else:
                            x1, y1, x2, y2 = symbol_box
                            label_id_ = int(symbol_label)
                            class_name = id_to_label.get(label_id_, 'Unknown')
                            
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)
                            # Add label text
                            # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, class_name, color='Red', fontsize=8)                              
                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'./output/vote_validation/faster_rcnn/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                            plt.close()
                            results.append((image_name, class_name, actual_label,'invalid', 'invalid', 'valid vote and invalid symbol',closet_distance))
                            print('Symbols' + str(id_to_label))
                            print('Vote: ' + class_name + "|" + "invalid|" + "valid vote and invalid symbol")                            

                    else:
                        # print(bounding_box, image_name, grid_cells)
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        label_id1 = int(pred_label.cpu().numpy())
                        class_name = id_to_label.get(label_id1, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                        
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        plt.savefig(f'./output/vote_validation/faster_rcnn/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=300)
                        plt.close()
                        results.append((image_name, class_name, actual_label, 'invalid','invalid', 'stamp not inside cell1','Nan'))
                        print('Symbols' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "stamp not inside cell")
                        # plt.close()

                elif label_id == invalid_stamp_id:
                    # print(label_id, image_name)
                    
                    if is_stamp_valid(bounding_box, grid_cells):                        
                        x1, y1, x2, y2 = bounding_box                        
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        label_id2 = int(pred_label.cpu().numpy())
                        class_name = id_to_label.get(label_id2, 'Unknown')
                        ax.text(x1, y1, class_name, color='blue', fontsize=8)

                        # results.append((image_name, class_name, 'invalid', 'invalid stamp', "nan"))
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        plt.savefig(f'./output/vote_validation/faster_rcnn/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=300)
                        plt.close()
                        results.append((image_name, class_name, actual_label, 'invalid','invalid', 'invalid stamp', "nan"))
                        print('Symbols' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "invalid stamp")

                    else:
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        label_id3 = int(pred_label.cpu().numpy())
                        class_name = id_to_label.get(label_id3, 'Unknown') 
                        ax.text(x1, y1, class_name, color='blue', fontsize=8)                      
                        
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization2
                        plt.savefig(f'./output/vote_validation/faster_rcnn/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        plt.close()
                        results.append((image_name, class_name, actual_label, 'invalid','invalid','Stamp not inside cell','nan'))
                        print('Symbols' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "Stamp not inside cell")

                elif all(x not in label_ids for x in [15, 40]):
                    x1, y1, x2, y2 = bounding_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    label_id3 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id3, 'Unknown') 
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                    
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization
                    plt.savefig(f'./output/vote_validation/ /no_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0, dpi=600)
                    plt.close()
                    results.append((image_name, 'no vote',actual_label,'invalid', 'no vote', 'no vote','nan'))
                    print('Symbols' + str(id_to_label))
                    print('Vote: ' + 'stamp not detected' + "|" + "invalid|" + "no stamp")  
                    break

                elif all(x in label_ids for x in [15, 40]):
                    x1, y1, x2, y2 = bounding_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    label_id3 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id3, 'Unknown') 
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'./output/vote_validation/faster_rcnn/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0, dpi=600)
                    plt.close()
                    results.append((image_name, 'multiple stamp detected', actual_label,'invalid', 'invalid', 'multi stamp','nan'))
                    print('Symbols' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break 
                
                elif total_count_valid_invalid > 1:
                    x1, y1, x2, y2 = bounding_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    label_id3 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id3, 'Unknown') 
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'./output/vote_validation/faster_rcnn/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    results.append((image_name, 'multiple stamp detected',actual_label,'invalid', 'invalid', 'multi stamp','nan'))
                    print('Symbols' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break

                elif count_15 > 1:
                    x1, y1, x2, y2 = bounding_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    label_id3 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id3, 'Unknown') 
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                    x1, y1, x2, y2 = bounding_box                        
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    # Add label text
                    # label_text = f"{pred_label}"  
                    label_id2 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id2, 'Unknown')
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'./output/vote_validation/faster_rcnn/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    results.append((image_name, 'multiple stamp detected',actual_label,'invalid', 'invalid', 'multi stamp','nan'))
                    print('Symbols' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break
                
                elif count_40 > 1:  
                    x1, y1, x2, y2 = bounding_box
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    label_id3 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id3, 'Unknown') 
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)  
                    x1, y1, x2, y2 = bounding_box                        
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                    ax.add_patch(rect)
                    # Add label text
                    # label_text = f"{pred_label}"  
                    label_id2 = int(pred_label.cpu().numpy())
                    class_name = id_to_label.get(label_id2, 'Unknown')
                    ax.text(x1, y1, class_name, color='blue', fontsize=8)                 
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'./output/vote_validation/faster_rcnn/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    results.append((image_name, 'multiple stamp detected',actual_label,'valid', 'invalid', 'multi stamp','nan'))
                    print('Symbols' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break

        #   Symbols: 1-Tree 2-Sun …. and Vote - Tree|Invalid|No stamp.                    
        df = pd.DataFrame(results, columns=['Image Id','Vote Symbol', 'Actual', 'Predicted','Valid', 'Remarks','closet_distance'])
        df.to_excel('./output/vote_results/vote_results_faster.xlsx')


    # def iou(self, boxA, boxB):
    #     """Compute the Intersection Over Union (IoU) of two bounding boxes."""
    #     # Determine the coordinates of the intersection rectangle
    #     xA = max(boxA[0], boxB[0])
    #     yA = max(boxA[1], boxB[1])
    #     xB = min(boxA[2], boxB[2])
    #     yB = min(boxA[3], boxB[3])

    #     # Compute the area of intersection rectangle
    #     interArea = max(0, xB - xA) * max(0, yB - yA)

    #     # Compute the area of both the prediction and ground-truth rectangles
    #     boxAArea = (boxA[2] - boxA[0]) * (boxA[3]- boxA[1])
    #     boxBArea = (boxB[2] - boxB[0]) * (boxB[3]- boxB[1])


    #     # Compute the IoU
    #     iou = interArea / float(boxAArea + boxBArea - interArea)
    #     return iou
