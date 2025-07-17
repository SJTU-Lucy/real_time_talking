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


def run(SDK, sr=16000, chunk=6400):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sr,
                    input=True,
                    frames_per_buffer=chunk)
    end = False
    start_time = time.time()
    SDK.set_zero_time(start_time)
    while not end:
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if len(audio_data) < chunk:
            audio_data = np.pad(audio_data, (0, chunk - len(audio_data)), mode="constant")
            end = True
        # emotion id "neutral": 0, "angry": 1, "happy": 2, "song": 3
        identity_id = 0
        emotion_id = 1
        SDK.wav_process_queue.put([audio_data, identity_id, emotion_id])

        if SDK.worker_exception:
            raise SDK.worker_exception
        if keyboard.is_pressed('esc'):
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    time.sleep(1)

    SDK.close()


if __name__ == "__main__":
    output_path = "./result/output.txt"
    SDK = StreamSDK(output_path, device, clip_time=0.2)
    SDK.setup()
    SDK.load_weight("assets/mute.wav")
    run(SDK)
