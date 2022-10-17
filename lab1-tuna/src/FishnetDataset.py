import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from config import CLASSES, RESIZE_TO


class FishnetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels.query('label_l2 == "YFT"')
        self.img_dir = img_dir
        self.transform = transform

        self.image_boxes = {}

        for i in range(len(self.img_labels)):
            filename = self.img_labels.iloc[i, 0]
            x_min = self.img_labels.iloc[i, 2]
            x_max = self.img_labels.iloc[i, 3]
            y_min = self.img_labels.iloc[i, 4]
            y_max = self.img_labels.iloc[i, 5]
            label = self.img_labels.iloc[i, 6]
            if filename in self.image_boxes:
                self.image_boxes[filename].append([x_min, y_min, x_max, y_max, label])
            else:
                self.image_boxes[filename] = [[x_min, y_min, x_max, y_max, label]]
        self.image_boxes = list(self.image_boxes.items())

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_boxes[idx][0] + ".jpg")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (RESIZE_TO, RESIZE_TO))
        image_resized /= 255.0

        # get the height and width of the image
        image_width = image.shape[1]
        image_height = image.shape[0]

        params = self.image_boxes[idx][1]
        boxes = []
        labels = []
        # areas = []
        for param in params:
            xmin_final = (param[0] / image_width) * RESIZE_TO
            xmax_final = (param[2] / image_width) * RESIZE_TO
            ymin_final = (param[1] / image_height) * RESIZE_TO
            ymax_final = (param[3] / image_height) * RESIZE_TO
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
            labels.append(CLASSES.index(param[4]))
            # areas.append((xmax_final - xmin_final) * (ymax_final - ymin_final))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {"boxes": boxes, "labels": labels, "area": area}

        if self.transform is not None:
            # img, target = self.transform(image, target)
            sample = self.transform(image=image_resized,
                                     bboxes=target['boxes'],
                                     labels=labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return image_resized, target
