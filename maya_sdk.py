import threading
import queue
import time
import torch
import logging
import socket
from model.net import EncoderTransformer
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
    def __init__(self, device):
        self.fun_socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.SPEED_PLAY = float(1.0 / 30)
        self.device = device
        self.wav_process = FeaturesConstructor()
        self.audio2face = EncoderTransformer(out_dim=51)
        self.audio2face.to(self.device)
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

    def load_weight(self):
        self.audio2face.load_state_dict(torch.load("weights/embed_50_model.pth", map_location=self.device))
        self.audio2face.eval()

        with torch.no_grad():
            dummy_audio = torch.randn(1, 3200).to(self.device)
            dummy_seq_len = torch.Tensor([6]).int().to(self.device)
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
            duration = time.perf_counter() - start
            logging.info(f"[Time] wav_process took {duration:.4f}s")
            # logging.info("processed wav data: %s %s", audio.shape, seq_len)
            res = [audio, seq_len, identity, emotion]
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
            audio, seq_len, identity, emotion = item
            logging.info(f"[Time] audio2face input {time.time() - self.zero_time:.4f}s")
            result = self.audio2face(audio, seq_len, identity, emotion)
            result = result.squeeze()
            result = result.detach().cpu().numpy()
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
