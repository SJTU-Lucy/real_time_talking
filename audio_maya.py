import scipy.io.wavfile as wavfile
import os
import time
import numpy as np
import random
import logging
import torch
import librosa
from maya_sdk import MayaSDK
from api_audio import AudioPlay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
logging.basicConfig(level=logging.DEBUG)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def run(SDK, wav_file, chunk=3200):
    raw_wav, _ = librosa.load(wav_file, sr=16000)
    gap_time = chunk / 16000
    player = AudioPlay()
    player.play_wav_file_thread(wav_path)
    next_sample_time = time.time() + gap_time
    sample_idx = 0
    end = False
    start_time = time.time()
    SDK.set_zero_time(start_time)
    while not end:
        now = time.time()
        if now >= next_sample_time:
            logging.info(f"[Time] Sample: %s", now - start_time)
            audio_data = raw_wav[sample_idx:sample_idx + chunk]
            if len(audio_data) < chunk:
                audio_data = np.pad(audio_data, (0, chunk - len(audio_data)), mode="constant")
                end = True
            next_sample_time += gap_time
            sample_idx += chunk
            # emotion id "neutral": 0, "angry": 1, "happy": 2, "song": 3
            identity_id = 1
            emotion_id = 2
            SDK.wav_process_queue.put([audio_data, identity_id, emotion_id])
        else:
            time.sleep(0.001)
        if SDK.worker_exception:
            raise SDK.worker_exception
    end_time = time.time()
    logging.info(f"[Time] Duration Time: %s", end_time - start_time)
    time.sleep(1)
    SDK.close()


if __name__ == "__main__":
    wav_path = "assets/speaker1_angry_CH_23_iPhone01.wav"
    SDK = MayaSDK(device, clip_time=0.2)
    SDK.setup()
    SDK.load_weight("assets/context.wav")
    time.sleep(2)
    run(SDK, wav_path, chunk=3200)
