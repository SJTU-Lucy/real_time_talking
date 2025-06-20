import os
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split
from transformers import Wav2Vec2Processor
import torch
import librosa
import numpy as np

label_dim = 174
emotion_id = {"ang": 0, "dis": 1, "fea": 2, "hap": 3, "neu": 4, "sad": 5, "sur": 6}


class AudioDataProcessor:
    def __init__(self, sampling_rate=16000) -> None:
        # self._processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        # self._processor = Wav2Vec2Processor.from_pretrained("C:/Users/Chang Liu/Desktop/wav2vec2-base-960h",
        #                                                     local_files_only=True)
        self._processor = Wav2Vec2Processor.from_pretrained("C:/Users/18158/Desktop/EmoFace/wav2vec2-base-960h",
                                                            local_files_only=True)
        self._sampling_rate = sampling_rate

    def run(self, audio):
        input_values = np.squeeze(self._processor(audio, sampling_rate=self._sampling_rate).input_values)
        return input_values

    @property
    def sampling_rate(self):
        return self._sampling_rate


class FeaturesConstructor:
    def __init__(self, audio_max_duration=60):
        self._audio_max_duration = audio_max_duration
        self._audio_data_processor = AudioDataProcessor()
        self._audio_sampling_rate = self._audio_data_processor.sampling_rate
        self._fps = 60

    def __call__(self,  audio):
        audio_data = self._audio_data_processor.run(audio)
        seq_len = int(len(audio_data) / self._audio_sampling_rate * self._fps)
        label_data = np.zeros((seq_len, label_dim))
        feature_chunk = [audio_data, label_data]
        return feature_chunk
