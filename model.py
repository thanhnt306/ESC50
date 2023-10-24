import torch.nn as nn
from torchvision import models
from torchsummary import summary


class EscClassificator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(1, 1, 9, 5, 2),
            nn.ReLU(),
            nn.Conv1d(1, 1, 9, 5, 2),
            nn.ReLU(),
            nn.Conv1d(1, 1, 4, 5, 2),
            nn.ReLU(),
            nn.Conv1d(1, 1, 4, 2, 1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(882, 660),
            nn.ReLU(),
            nn.Linear(660, 440),
            nn.ReLU(),
            nn.Linear(440, 220),
            nn.ReLU(),
            nn.Linear(220, 110),
            nn.ReLU(),
            nn.Linear(110, 50),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1d(input_data)
        x = self.flatten(x)
        logits = self.fc(x)
        predictions = self.softmax(logits)
        return predictions


if __name__ == "__main__":
    nn = EscClassificator()
    summary(nn, (1, 220500))
