import torch.nn as nn
import torch.nn.functional as F


# Пример нейронной сети
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 128, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.3)

        self.conv2 = nn.Conv2d(128, 224, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout(p=0.4)

        self.fc3 = nn.Linear(224 * 4 * 4, 64)
        self.drop3 = nn.Dropout(p=0.4)

        self.fc4 = nn.Linear(64, 32)
        self.drop4 = nn.Dropout(p=0.4)

        self.fc5 = nn.Linear(32, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))

        x = x.view(-1, 224 * 4 * 4)

        x = self.drop3(F.relu(self.fc3(x)))
        x = self.drop4(F.relu(self.fc4(x)))

        x = self.softmax(self.fc5(x))

        return x
