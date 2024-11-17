import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import numpy as np
import random
import pandas as pd
import os
from PIL import Image
# from utils.utils import parse_args, load_config, torch_nms
from modules.symbol_detection.faster_rcnn.utils.faster_rcnn_utils import torch_nms
from modules.symbol_detection.faster_rcnn.entity.config_entity import EvaluationConfig

import warnings
warnings.filterwarnings('ignore', 'Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)')


class VisualizePrediction():

    def __init__(self):
        pass
    
    def random_color(self):
    # Generate random RGB components
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Convert RGB values to a hex color code
        return f"#{r:02x}{g:02x}{b:02x}" 
    
    def wrap_text(self, text, max_width, font_properties, ax):
        words = text.split('_')
        wrapped_lines = []
        current_line = ""
        for word in words:
            test_line = current_line + word + "_"
            text_bbox = ax.text(0, 0, test_line, fontdict=font_properties).get_window_extent(renderer=plt.gcf().canvas.get_renderer())
            if text_bbox.width > max_width:
                if current_line:
                    wrapped_lines.append(current_line.strip('_'))
                current_line = word + "_"
            else:
                current_line = test_line
        if current_line:
            wrapped_lines.append(current_line.strip('_'))
        return wrapped_lines
    
    def visualize_predicted_images(self, test_images_path, test_set, predicted_labels, label_to_id):
                  
        for i, (test_data, label) in enumerate(zip(test_set, predicted_labels)):

            image_name = test_data[2]
            print(test_data,label)
            image = os.path.join(test_images_path, image_name)            
            image = Image.open(image)                       
            # img_np = image.permute(1, 2, 0).numpy()         
            id_to_label = {value: key for key, value in label_to_id.items()}
            
            fig, ax = plt.subplots(1) 
            ax.imshow(image) 
            
            # actual_labels = test_data[1]['labels']  
            # actual_bounding_box = test_data[1]['boxes']
            # # image_name1 = test_data[2]

            boxes = label[0]['boxes'] 
            labels = label[0]['labels']
            scores = label[0]['scores']
            image_name = label[2]   
    
            indices = torch_nms(boxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )      

            final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices]             

            for box, score, pred_label in zip(final_boxes, final_scores, final_labels):                
                
                box1 = box.cpu().numpy() 
                label_id = int(f"{pred_label}")
                class_name = id_to_label.get(label_id, 'Unknown') 
                x1, y1, x2, y2 = box1 
                font_properties = {'family': 'Times New Roman', 'size': 4}
                grid_width = 372
                if label_id == 15 or label_id == 40:

                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5, edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)                   

                    # Wrap the text
                    wrapped_lines = self.wrap_text(class_name, grid_width, font_properties, ax)

                    # Add text with background rectangles
                    for i, line in enumerate(wrapped_lines):
                        print('test_stamp', i)  # Change: Added print statement for debugging
                        # Calculate the y position for each line
                        y_pos = y1 - i * (font_properties['size'] + 1)  # Adjust line spacing as needed
                        
                        # Create text with alpha=0 for initial rendering to avoid showing at top left corner
                        text = ax.text(x1, y_pos, line, color='#FFFFE0', fontdict=font_properties, alpha=0)  # Change: Added alpha=0
                        # fig.canvas.draw()
                        text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
                        text_bb = text_bb.transformed(ax.transData.inverted())
                        
                        # Now, create the text with actual color and set alpha back to 1
                        text.remove()  # Change: Remove initial text with alpha=0
                        text = ax.text(x1, y_pos, line, color='#FFFFE0', fontdict=font_properties)  # Change: Added final text rendering

                        # Add a rectangle with the same dimensions behind the text
                        highlight = patches.Rectangle((text_bb.x0, text_bb.y0), text_bb.width, text_bb.height, 
                                                    color='#006400', alpha=1, zorder=text.get_zorder()-1)
                        ax.add_patch(highlight)
                else:
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    wrapped_lines = self.wrap_text(class_name, grid_width, font_properties, ax)
                    for i, line in enumerate(wrapped_lines):
                        print('test_symbol', i)  # Change: Added print statement for debugging
                        # Calculate the y position for each line
                        y_pos = y1 - i * (font_properties['size'] + 1)  # Adjust line spacing as needed
                        
                        # Create text with alpha=0 for initial rendering to avoid showing at top left corner
                        text = ax.text(x1, y_pos, line, color='#FFB6C1', fontdict=font_properties, alpha=0)  # Change: Added alpha=0
                        fig.canvas.draw()
                        text_bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
                        text_bb = text_bb.transformed(ax.transData.inverted())

                        # Now, create the text with actual color and set alpha back to 1
                        text.remove()  # Change: Remove initial text with alpha=0
                        text = ax.text(x1, y_pos, line, color='#FFB6C1', fontdict=font_properties)  # Change: Added final text rendering

                        # Add a rectangle with the same dimensions behind the text
                        highlight = patches.Rectangle((text_bb.x0, text_bb.y0), text_bb.width, text_bb.height, 
                                                    color ='#4B0082', alpha=1, zorder=text.get_zorder()-1)
                        ax.add_patch(highlight)

                # Add label text
                # label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                            
         
            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            plt.savefig(f'../../../../../output/visualization/faster_rcnn/{image_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)           
            plt.close()
            
                    
          
    def visualize_train_set(self, train_labels, label_to_id):
           
            image = train_labels[0][0]
            boxes = train_labels[0][3]['boxes'] 
            labels = train_labels[0][3]['labels']
            # scores = train_labels[1]['scores']
            image_name = train_labels[0][2]
            img_np = image.permute(1, 2, 0).numpy()         
            id_to_label = {value: key for key, value in label_to_id.items()}
            fig, ax = plt.subplots(1) 
            ax.imshow(img_np)       
                         
                      

            # indices = self.apply_nms(boxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )           

            for box, label in zip(boxes, labels):

                # score1 = score.cpu().numpy()                      
                box1 = box.cpu().numpy()
                label_id = int(f"{label}")
                class_name = id_to_label.get(label_id, 'Unknown')
                x1, y1, x2, y2 = box1

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5, edgecolor='blue', facecolor='none')
                ax.add_patch(rect)

                # Add label text
                # label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                ax.text(x1, y1, class_name, color='blue', fontsize=9) 
            
            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            # plt.savefig(f'../../../outputdir/{image_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)  
            plt.show()         
            plt.close()
     