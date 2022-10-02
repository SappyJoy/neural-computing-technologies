import os.path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image


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
        self.image_boxes = self.image_boxes.items()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, +".jpg")
        image = read_image(img_path)
        params = self.image_boxes[idx]
        boxes = []
        labels = []
        areas = []
        for param in params:
            boxes.append(torch.as_tensor(params[0:3]))
            labels.append(params[4])
            areas.append((param[2] - param[0]) * (param[3] - param[1]))

        # targets = []
        target = {"box": box, "label": label, "area": area}
        # targets.append(target)
        # target["area"] = area
        if self.transform:
            image = self.transform(image)

        return image, boxes, labels
