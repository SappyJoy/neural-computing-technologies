import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter  # TensorBoard support
from torchvision import datasets, transforms

tb = SummaryWriter(comment="Run simple cnn on mnist")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]),
    download=True,
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]),
)

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1)
