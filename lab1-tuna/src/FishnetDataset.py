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

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0] + ".jpg")
        image = read_image(img_path)
        x_min = self.img_labels.iloc[idx, 2]
        x_max = self.img_labels.iloc[idx, 3]
        y_min = self.img_labels.iloc[idx, 4]
        y_max = self.img_labels.iloc[idx, 5]
        box = torch.as_tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        label = self.img_labels.iloc[idx, 7]
        area = (box[:, 3] - box[:, 1]) * (box[:, 2] - box[:, 0])

        target = {}
        target["box"] = box
        target["label"] = label
        target["area"] = area
        if self.transform:
            image = self.transform(image)

        return image, label
