from torch import nn
from torch.nn import functional as F
import torch


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input: [1, 28, 28], Output: [32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input: [32, 28, 28], Output: [64, 28, 28]
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # Input: [64, 28, 28], Output: [128, 28, 28]

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)  # After max pooling 3 times, feature map size is 3x3
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Input shape: [batch_size, 1, 28, 28]

        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.conv1(x))  # Output: [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output: [batch_size, 32, 14, 14]

        x = F.relu(self.conv2(x))  # Output: [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output: [batch_size, 64, 7, 7]

        x = F.relu(self.conv3(x))  # Output: [batch_size, 128, 7, 7]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Output: [batch_size, 128, 3, 3]

        # Flatten before the fully connected layers
        x = torch.flatten(x, start_dim=1)  # Flatten to [batch_size, 128*3*3]

        # Fully connected layers with dropout
        x = self.dropout(F.relu(self.fc1(x)))  # Output: [batch_size, 128]
        x = self.dropout(F.relu(self.fc2(x)))  # Output: [batch_size, 64]

        # Output layer (no activation; handled by loss function)
        x = self.fc3(x)  # Output: [batch_size, 10]
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
