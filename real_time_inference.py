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

from stream_pipeline import StreamSDK

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

    while not keyboard.is_pressed('left ctrl'):
        continue

    while keyboard.is_pressed('left ctrl'):
        data = stream.read(chunk)
        audio_data = np.frombuffer(data, dtype=np.int16)
        if len(audio_data) < chunk:
            audio_data = np.pad(audio_data, (0, chunk - len(audio_data)), mode="constant")
        SDK.wav_process_queue.put(audio_data)
        logging.info("detecting audio: %s", audio_data.shape)

        if SDK.worker_exception:
            raise SDK.worker_exception

    stream.stop_stream()
    stream.close()
    p.terminate()
    time.sleep(10000)

    SDK.close()


if __name__ == "__main__":
    output_path = "./result/output.txt"
    SDK = StreamSDK(output_path, device)
    SDK.setup()
    run(SDK)
