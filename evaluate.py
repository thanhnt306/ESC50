import numpy
import torch
import torch.optim

from dataLoader import ESC50Dataset
from model import EscClassificator
from torch.utils.data import DataLoader

META_FILE = 'ESC-50-master/meta/esc50.csv'
AUDIO_PATH = 'ESC-50-master/audio/'


def predict_result(model, validate_data_loader):
    num_of_true = 0
    model.eval()
    with torch.no_grad():
        for input, target in validate_data_loader:
            predictions = model(input)
            predicted_index = predictions[0].argmax(0)
            predicted_index = predicted_index
            a = 1
            if predicted_index.item() == target:
                num_of_true += 1
    return num_of_true


if __name__ == "__main__":
    model = EscClassificator()
    state_dict = torch.load("EscClassificator.pth")
    model.load_state_dict(state_dict)

    esc50_val = ESC50Dataset(
        csv_file_path=META_FILE,
        audio_path=AUDIO_PATH,
        training=False
    )

    accuracy = predict_result(model, esc50_val)
    print('accuracy:', accuracy)
