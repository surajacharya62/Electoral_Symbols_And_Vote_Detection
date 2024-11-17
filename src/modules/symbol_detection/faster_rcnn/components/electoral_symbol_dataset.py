import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


class ElectoralSymbolDataset(Dataset):
    def __init__(self, image_path, image_name, annotation_path, label, transforms, is_single_image=False):
        self.image_path = image_path
        # self.train_or_test_path = train_or_test_path
        self.transforms = transforms
        self.label_to_id = label
        self.is_single_image = is_single_image
        self.annotation_path = annotation_path        

        if is_single_image:
            # Handle single image case
            self.imgs_list = [image_name]
            print("Length of dataset:", len(self.imgs_list))  # Single image in the list
            # annotations_path = os.path.join(self.annotation_path)
            if os.path.exists(annotation_path):
                self.df = pd.read_csv(annotation_path)
            else:
                # If annotations for a single image are to be handled differently
                raise FileNotFoundError("Annotations file not found for the single image mode.")
        else:
            
            # Handle directory case
            self.df = pd.read_csv(self.annotation_path)
            self.imgs_list = sorted(os.listdir(image_path))
            print("Length of dataset:", len(self.imgs_list)) 

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        
        if self.is_single_image:
            img_name = img_name  
            img_path = os.path.join(self.image_path, img_name)
            
        else:
            img_path = os.path.join(self.image_path, img_name)

        img = Image.open(img_path).convert('RGB')
        filtered_rows = self.df[self.df['image_id'] == img_name]
        boxes = filtered_rows[['x1', 'y1', 'x2', 'y2']].values.astype('float32')
        labels = filtered_rows['label'].apply(lambda x: self.label_to_id[x]).values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = labels.astype(np.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, image_id, img_name, target

    def __len__(self):
        return len(self.imgs_list)


