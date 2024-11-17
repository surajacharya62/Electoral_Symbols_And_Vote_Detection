import argparse
import torch 
import yaml
import pandas as pd
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser(description="Electoral Symbol and Vote Detection and  Validation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    args = parser.parse_args()
    return args


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def torch_nms(original_boxes, original_scores, iou_threshold=0.5):    
    keep = torch.ops.torchvision.nms(original_boxes, original_scores, iou_threshold)
    # print(keep)
    return keep 


def label_to_id(label_path):
    df = pd.read_csv(label_path)
    label_to_id = {label: i for i, label in enumerate(df['label'].unique(),1)}
    sorted_labels = sorted(label_to_id)    
    label_to_id = {label: i for i, label in enumerate(sorted_labels, 1)}    
    return label_to_id 

def get_transform(train):
    transforms = []
    if train:
        transforms = T.Compose([
                # T.Resize((224, 224)), 
                T.ToTensor(),
                # T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
            ])
    else:
        transforms = T.Compose([
                # T.Resize((224, 224)),  
                T.ToTensor(),
                # T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
            ])

    return transforms


def collate_fn(batch):
    return tuple(zip(*batch))

def calculate_iou(boxA, boxB):
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


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    # Slicing idx:idx+1 in order to keep tensor dimensionality
    # Doing ... in indexing if there would be additional dimensions
    # Like for Yolo algorithm which would have (N, S, S, 4) in shape
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Need clamp(0) in case they do not intersect, then we want intersection to be 0
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)    


def nms(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [ 
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or self.intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns):
    mt, mb, ml, mr = margins
    ballot_width, ballot_height = ballot_size
    symbol_width, symbol_height = symbol_size

    # Calculate cell size based on the symbol size being half the width of the cell
    cell_width = symbol_width * 2  # Ensure each cell is double the width of the symbol
    cell_height = symbol_height    # Height of the cell matches the height of the symbol

    # Initialize grid cells list
    grid_cells = []

    # Calculate the starting y-coordinate of the grid
    header_box_bottom = mt
    grid_start_y = header_box_bottom  # Additional offset for the header box

    # Calculate the starting x-coordinate of the grid
    grid_start_x = ml

    # Generate grid cells based on calculated dimensions
    for row_idx in range(rows):
        for col_idx in range(columns):
            x1 = grid_start_x + col_idx * cell_width
            y1 = grid_start_y + row_idx * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            grid_cells.append((x1, y1, x2, y2))

    return grid_cells 


def is_stamp_valid(stamp_box, grid_cells, tolerance=10):
    
    for cell_box in grid_cells:
        # Expand the cell box by the tolerance value
        adjusted_cell_box = (
            cell_box[0] - tolerance,  # left
            cell_box[1] - tolerance,  # top
            cell_box[2] + tolerance,  # right
            cell_box[3] + tolerance   # bottom
        )
        if (stamp_box[0] >= adjusted_cell_box[0] and
            stamp_box[1] >= adjusted_cell_box[1] and
            stamp_box[2] <= adjusted_cell_box[2] and
            stamp_box[3] <= adjusted_cell_box[3]):
            return True
    return False


def center(box):
        """Calculate the center point of a bounding box."""
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return (x_center, y_center)


def calculate_edge_distance(stamp_box, symbol_box):
    """Calculate the horizontal and vertical distances between the stamp and symbol."""
    horizontal_distance = max(0, stamp_box[0] - symbol_box[2]) 
        # Stamp to the right of the symbol
    # print(stamp_box[0],symbol_box[2])
    vertical_distance = 0  # Default to 0 if vertically aligned or overlapping
    if stamp_box[3] < symbol_box[1]: 
            # Stamp above symbol
    
        vertical_distance = symbol_box[1] - stamp_box[3]
    elif symbol_box[3] < stamp_box[1]:  # Stamp below symbol
        vertical_distance = stamp_box[1] - symbol_box[3]
    
    return horizontal_distance, vertical_distance


def is_stamp_for_symbol(stamp_box, symbol_boxes, image_name):
    """Determine if a stamp is for a symbol based on overlap or proximity."""
    closest_distance = float('inf') 
    closest_symbol_label = None
    for symbol_box, symbol_label, score in symbol_boxes:
        # print(symbol_label)
        symbol_label = symbol_label.cpu().numpy()
        symbol_box = symbol_box.cpu().numpy()
        # symbol_box = [round(coordinate) for coordinate in symbol_box]
        
        if symbol_label not in [40, 15]:
            # print(True, symbol_label)
            if calculate_iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
                return True, symbol_label, symbol_box, closest_distance
            else:  # Check for proximity
                dist = ((center(symbol_box)[0] - center(stamp_box)[0]) ** 2 + 
                        (center(symbol_box)[1] - center(stamp_box)[1]) ** 2) ** 0.5
                if dist <= closest_distance:
                    closest_distance = dist
                    # print(closest_distance)
                    closest_symbol_label = symbol_label
                    closest_symbol_box = symbol_box
    # Check if the closest symbol is within the acceptable threshold distance
    # if closest_distance < proximity_threshold:
    #     return True, closest_symbol_label, symbol_box
    adjusted_proximity_threshold = 378 - 189  # Example adjustment
    
    if closest_distance <= adjusted_proximity_threshold:
        # print("True" ,closest_distance, image_name)
        return True, closest_symbol_label, closest_symbol_box, closest_distance
    else:
        # print("False",closest_distance, image_name)
        
        return False, closest_symbol_label, closest_symbol_box, closest_distance
    

def proximity(boxA, boxB, threshold):
    """Check if two boxes are within a certain threshold distance."""
    centerA = center(boxA)
    centerB = center(boxB)
    distance = ((centerB[0] - centerA[0]) ** 2 + (centerB[1] - centerA[1]) ** 2) ** 0.5
    return distance < threshold


