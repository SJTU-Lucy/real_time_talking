import threading
import queue
import time
import torch
import logging
import socket
import librosa
import numpy as np
from scipy import signal
from transformers import Wav2Vec2Processor
from model.net import HuBERTFeatureExtractor
from model.data import FeaturesConstructor

logging.basicConfig(level=logging.DEBUG)


bs_list = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight',
           'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight',
           'eyeWideLeft', 'eyeWideRight', 'eyeLookOutLeft', 'eyeLookOutRight',
           'eyeLookUpLeft', 'eyeLookUpRight', 'browDownLeft', 'browDownRight',
           'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'jawOpen',
           'mouthClose', 'jawLeft', 'jawRight', 'jawForward', 'mouthUpperUpLeft',
           'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
           'mouthRollUpper', 'mouthRollLower', 'mouthSmileLeft', 'mouthSmileRight',
           'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft',
           'mouthStretchRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPressLeft',
           'mouthPressRight', 'mouthPucker', 'mouthFunnel', 'mouthLeft', 'mouthRight',
           'mouthShrugLower', 'mouthShrugUpper', 'noseSneerLeft', 'noseSneerRight',
           'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight']


class MayaSDK:
    def __init__(self, device, clip_time=0.2):
        self.fun_socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fps = 60
        self.sr = 16000
        self.context_len = 48000
        self.clip_len = int(clip_time * self.sr)
        self.window_frames = int(clip_time * self.fps)
        self.SPEED_PLAY = float(1.0 / self.fps)
        self.device = device
        self.wav_process = FeaturesConstructor()
        self.audio2face = HuBERTFeatureExtractor(out_dim=51).to(self.device)
        self.audio2face.to(self.device)
        self.prev_output = None
        self.zero_time = None

    def setup(self):
        self.fun_socket_send.connect(('127.0.0.1', 50001))
        # ======== Setup Worker Threads ========
        QUEUE_MAX_SIZE = 100

        self.worker_exception = None
        self.stop_event = threading.Event()

        self.wav_process_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.audio2face_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
        self.write_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)

        self.thread_list = [
            threading.Thread(target=self.wav_process_worker),
            threading.Thread(target=self.audio2face_worker),
            threading.Thread(target=self.write_worker)
        ]

        for thread in self.thread_list:
            thread.start()

        self.f_num = 0

    def load_weight(self, audio_path):
        self.audio2face.load_state_dict(torch.load("weights/hubert_50_model.pth", map_location=self.device))
        self.audio2face.eval()

        self._processor = Wav2Vec2Processor.from_pretrained("C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h",
                                                            local_files_only=True)
        speech_array, sampling_rate = librosa.load(audio_path, sr=self.sr)
        warmup_audio = np.squeeze(self._processor(speech_array, sampling_rate=sampling_rate).input_values)
        warmup_audio = torch.from_numpy(warmup_audio[-self.context_len:]).unsqueeze(0)

        self.buffer = warmup_audio.to(self.device)

        with torch.no_grad():
            dummy_audio = warmup_audio.to(self.device)
            dummy_seq_len = torch.Tensor([warmup_audio.shape[1] * 60 // 16000]).int().to(self.device)
            dummy_identity = torch.zeros(1).long().to(self.device)
            dummy_emotion = torch.zeros(1).long().to(self.device)
            _ = self.audio2face(dummy_audio, dummy_seq_len, dummy_identity, dummy_emotion)

    def set_zero_time(self, time):
        self.zero_time = time

    def wav_process_worker(self):
        try:
            self._wav_process_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
            print("set stop event")

    def _wav_process_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.wav_process_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break
            start = time.perf_counter()
            raw_wav, identity, emotion = item
            logging.info(f"[Time] wav_process input {time.time() - self.zero_time:.4f}s")
            feature_chunk = self.wav_process(raw_wav)
            audio, seq_len = feature_chunk
            audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            seq_len = torch.Tensor([seq_len]).int().to(self.device)
            emotion = torch.Tensor([emotion]).long().unsqueeze(0).to(self.device)
            identity = torch.Tensor([identity]).long().unsqueeze(0).to(self.device)
            # fake context
            full_audio = torch.cat([self.buffer, audio], dim=1)
            full_seq_len = int(full_audio.shape[1] / self.sr * self.fps)
            total_len = full_audio.shape[1]
            if total_len > self.context_len:
                self.buffer = full_audio[:, -self.context_len:].detach()
            else:
                self.buffer = full_audio.detach()

            duration = time.perf_counter() - start
            logging.info(f"[Time] wav_process took {duration:.4f}s")
            # logging.info("processed wav data: %s %s", full_audio.shape, full_seq_len)
            res = [full_audio, full_seq_len, identity, emotion]
            self.audio2face_queue.put(res)

    def audio2face_worker(self):
        try:
            self._audio2face_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def _audio2face_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.audio2face_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.write_queue.put(None)
                break
            start = time.perf_counter()
            full_audio, full_seq_len, identity, emotion = item
            logging.info(f"[Time] audio2face input {time.time() - self.zero_time:.4f}s")
            out = self.audio2face(full_audio, full_seq_len, identity, emotion)
            current_output = out[:, -self.window_frames:].squeeze(0).detach().cpu().numpy()

            if self.prev_output is None:
                current_output_cat = current_output
            else:
                current_output_cat = np.concatenate([self.prev_output, current_output], axis=0)

            current_output_cat = current_output_cat.T
            output = signal.savgol_filter(current_output_cat, window_length=5, polyorder=2, mode="nearest").T
            result = output[-self.window_frames:]

            self.prev_output = result

            duration = time.perf_counter() - start
            logging.info(f"[Time] audio2face took {duration:.4f}s")
            # logging.info("pred rig: %s", result.shape)
            self.write_queue.put(result)

    def write_worker(self):
        try:
            self._write_worker()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()

    def arkit_to_mel(self, arkit):
        blendshape_node = "BS_Node"

        mel_lines = []
        for name, value in zip(bs_list, arkit):
            mel_line = f'setAttr "{blendshape_node}.{name}" {value:.4f};'
            mel_lines.append(mel_line)

        mel_script = "\n".join(mel_lines)
        output = bytes(mel_script, encoding='utf-8')

        return output

    def _write_worker(self):
        is_first = True
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break
            start = time.perf_counter()

            arkits = item
            logging.info(f"[Time] writer input {time.time() - self.zero_time:.4f}s")

            if is_first:
                f_btime = time.time()
                is_first = False

            for i, arkit in enumerate(arkits):
                self.f_num += 1
                mel = self.arkit_to_mel(arkit)
                self.fun_socket_send.send(mel)
                target_time = self.SPEED_PLAY * self.f_num
                current_time = time.time() - f_btime
                # logging.info(f"{target_time:.4f}, {current_time:.4f}")
                sleep_time = target_time - current_time - 0.01
                if sleep_time > 0:
                    time.sleep(sleep_time)
            duration = time.perf_counter() - start
            logging.info(f"[Time] writer took {duration:.4f}s")

    def close(self):
        # flush frames
        self.wav_process_queue.put(None)
        self.audio2face_queue.put(None)
        self.write_queue.put(None)

        # Wait for worker threads to finish
        for thread in self.thread_list:
            thread.join()

        # Check if any worker encountered an exception
        if self.worker_exception is not None:
            raise self.worker_exception
