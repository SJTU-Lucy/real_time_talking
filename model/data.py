import os
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import Wav2Vec2Processor
import torch
import librosa
import numpy as np

class AudioDataProcessor:
    def __init__(self, sampling_rate=16000) -> None:
        self._processor = Wav2Vec2Processor.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                            local_files_only=True)
        self._sampling_rate = sampling_rate

    def run(self, audio):
        input_values = np.squeeze(self._processor(audio, sampling_rate=self._sampling_rate).input_values)
        return input_values

    @property
    def sampling_rate(self):
        return self._sampling_rate


class FeaturesConstructor:
    def __init__(self):
        self._audio_data_processor = AudioDataProcessor()
        self._audio_sampling_rate = self._audio_data_processor.sampling_rate
        self._fps = 60

    def __call__(self,  audio):
        audio_data = self._audio_data_processor.run(audio)
        seq_len = int(len(audio_data) / self._audio_sampling_rate * self._fps)
        feature_chunk = [audio_data, seq_len]
        return feature_chunk
