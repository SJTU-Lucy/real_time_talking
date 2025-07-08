import librosa
import math
import os
import time
import numpy as np
import random
import torch
import pyaudio
import keyboard
import logging

from stream_sdk import StreamSDK

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.DEBUG)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(SDK, wav_file, sr=16000, chunk=6400):
    raw_wav, _ = librosa.load(wav_file, sr=sr)
    for i in range(0, raw_wav.shape[0], chunk):
        audio_data = raw_wav[i:i+chunk]
        if len(audio_data) < chunk:
            audio_data = np.pad(audio_data, (0, chunk - len(audio_data)), mode="constant")
            break
        # emotion id "neutral": 0, "angry": 1, "happy": 2, "song": 3
        identity_id = 0
        emotion_id = 1
        SDK.wav_process_queue.put([audio_data, identity_id, emotion_id])
        logging.info("detecting audio: %s", audio_data.shape)

        if SDK.worker_exception:
            raise SDK.worker_exception
    SDK.close()


if __name__ == "__main__":
    output_path = "./result/output.txt"
    wav_path = "C:/Users/86134/Desktop/data/all_emotions/train/WAV/angry/common/angry_gd02_001_029.wav"
    SDK = StreamSDK(output_path, device)
    SDK.setup()
    SDK.load_weight()
    run(SDK, wav_path)
