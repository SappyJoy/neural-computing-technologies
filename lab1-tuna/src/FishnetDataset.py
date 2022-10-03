import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

from config import CLASSES


class FishnetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

        self.image_boxes = {}

        for i in range(len(self.img_labels)):
            filename = self.img_labels.iloc[i, 0]
            x_min = self.img_labels.iloc[i, 2]
            x_max = self.img_labels.iloc[i, 3]
            y_min = self.img_labels.iloc[i, 4]
            y_max = self.img_labels.iloc[i, 5]
            label = self.img_labels.iloc[i, 7]
            if filename in self.image_boxes:
                self.image_boxes[filename].append([x_min, y_min, x_max, y_max, label])
            else:
                self.image_boxes[filename] = [[x_min, y_min, x_max, y_max, label]]
        self.image_boxes = list(self.image_boxes.items())

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_boxes[idx][0] + ".jpg")
        image = read_image(img_path)
        params = self.image_boxes[idx][1]
        boxes = []
        labels = []
        areas = []
        for param in params:
            boxes.append(param[0:4])
            labels.append(CLASSES.index(param[4]))
            areas.append((param[2] - param[0]) * (param[3] - param[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": areas}

        if self.transform is not None:
            img, target = self.transform(image, target)

        return image, target
