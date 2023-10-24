import torchaudio
import torch
import pandas as pd
from torch.utils.data import Dataset

NUM_OF_CLASS = 50
META_FILE = 'ESC-50-master/meta/esc50.csv'
AUDIO_PATH = 'ESC-50-master/audio/'


class ESC50Dataset(Dataset):

    def __init__(self,
                 csv_file_path,
                 audio_path,
                 training=True):
        self.csv_file = pd.read_csv(csv_file_path)
        self.audio_path = audio_path
        self.is_for_training = training

    def __len__(self):
        if self.is_for_training:
            return len(self.csv_file) - 400
        else:
            return 400

    def __getitem__(self, index):
        if self.is_for_training:
            audio_file_path = self._get_audio_file_path(index)
            signal, sr = torchaudio.load(audio_file_path)
            label = self._get_audio_file_label(index)

            return signal, label
        else:
            audio_file_path = self._get_audio_file_path(index + 1600)
            signal, sr = torchaudio.load(audio_file_path)
            label = self._get_audio_file_label(index + 1600)

            return signal, label

    def _get_audio_file_path(self, item):
        return AUDIO_PATH + self.csv_file.iloc[item, 0]

    def _get_audio_file_label(self, item):
        return self.csv_file.iloc[item, 2]


if __name__ == '__main__':
    esc50 = ESC50Dataset(csv_file_path=META_FILE, audio_path=AUDIO_PATH, training=True)
