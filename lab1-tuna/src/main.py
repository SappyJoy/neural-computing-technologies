import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from FishnetDataset import FishnetDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
from torchvision import models

tb = SummaryWriter(comment="Run simple cnn on mnist")

# Device configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # My GeForce 660 isn't supported because it's too old
print(device)

train_data = FishnetDataset(
    annotations_file="../resources/labels/foid_labels_bbox_v012.csv",
    img_dir="../resources/images",
    transform=None,
)

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    print("label={}".format(label))
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.permute(1, 2, 0))
plt.show()

train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)

test_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)

cnn = models.alexnet(num_classes=10).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0015)

print(cnn)
