import numpy
import torch
import torch.optim

from dataLoader import ESC50Dataset
from model import EscClassificator
from torch.utils.data import DataLoader
from torch import nn

META_FILE = 'ESC-50-master/meta/esc50.csv'
AUDIO_PATH = 'ESC-50-master/audio/'
LEARNING_RATE = 0.01
BATCH_SIZE = 1600
EPOCHS = 100


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i + 1}")
        train_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


def train_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=256)
    return train_dataloader


if __name__ == "__main__":
    model = EscClassificator()

    # state_dict = torch.load("EscClassificator.pth")
    # model.load_state_dict(state_dict)

    esc50 = ESC50Dataset(
        csv_file_path=META_FILE,
        audio_path=AUDIO_PATH,
        training=True
    )
    train_dataloader = create_data_loader(esc50, BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train(model, train_dataloader, loss_fn, optimizer, device='cpu', epochs=EPOCHS)

    torch.save(model.state_dict(), "EscClassificator.pth")
    print("Trained feed forward net saved at EscClassificator.pth")



