import threading
import queue
import time
import numpy as np
import traceback
import random
from scipy import signal
import math
import torch
import logging

from model.net import EmoFace
from model.data import FeaturesConstructor

logging.basicConfig(level=logging.DEBUG)


class StreamSDK:
    def __init__(self, output_path, device):
        self.device = device
        self.wav_process = FeaturesConstructor()
        self.audio2face = EmoFace(emo_dim=7, out_dim=174).to(self.device)
        self.output_path = output_path

        # random blink and gaze params
        self.blink_dic = [0.95, 0.75, 0.55, 0.3, 0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 0.75, 0.9]
        self.blink_dur = 13
        self.mu = 3.518
        self.sigma = 0.532
        self.switch_dur = 5
        self.return_rate = 0.4
        self.interval_lower = 25
        self.interval_upper = 45
        self.radius_lower = 0.1
        self.radius_upper = 0.2


    def setup(self):
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

    def load_weight(self):
        self.audio2face.load_state_dict(torch.load("weights/1000_model.pth", map_location=self.device))
        self.audio2face.eval()

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
            raw_wav = item
            feature_chunk = self.wav_process(raw_wav)
            audio, label = feature_chunk
            emotion = np.array([4])
            audio, label, emotion = torch.from_numpy(audio).float(), torch.from_numpy(label).float(), torch.from_numpy(
                emotion).int()
            audio, label, emotion = audio.unsqueeze(0), label.unsqueeze(0), emotion.unsqueeze(0)
            audio, label, emotion = audio.to(self.device), label.to(self.device), emotion.to(self.device)
            duration = time.perf_counter() - start
            logging.info(f"[Time] wav_process took {duration:.4f}s")
            # logging.info("processed wav data: %s %s", audio.shape, label.shape)
            res = [audio, label, emotion]
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
            audio, label, emotion = item
            result = self.audio2face(audio, label, emotion)
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

    def _write_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.write_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                break
            start = time.perf_counter()
            pred_rig = item
            # add blink and gaze
            # pred_rig = self.add_blink(pred_rig)
            # pred_rig = self.add_gaze(pred_rig)
            duration = time.perf_counter() - start
            logging.info(f"[Time] writer took {duration:.4f}s")
            # logging.info("data ready to be write: %s", pred_rig.shape)

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


    def sample_interval(self):
        sampled_frequency = random.lognormvariate(self.mu, self.sigma)
        sampled_frequency = min(100, max(15, sampled_frequency))
        interval_frame = int(3600 / sampled_frequency)
        return interval_frame


    def add_blink(self, pred):
        interval_frame = self.sample_interval()
        blink_count = 0
        for i in range(pred.shape[0]):
            k = 1
            if interval_frame >= 0:
                interval_frame -= 1
            else:
                k = self.blink_dic[blink_count]
                blink_count += 1
                if blink_count == self.blink_dur:
                    interval_frame = self.sample_interval()
                    blink_count = 0
                # rig 37 and 106 stands for blinks
                pred[i][36] = 1 - (1 - pred[i][36]) * k
                pred[i][105] = 1 - (1 - pred[i][105]) * k
        return pred


    def add_gaze(self, pred):
        gaze_interval = random.randint(25, 45)
        switch_count = 0
        current_point = (0, 0)
        next_point = None
        for i in range(pred.shape[0]):
            if gaze_interval >= 0:
                gaze_interval -= 1
                coordinate = current_point
            else:
                if switch_count == 0:
                    if random.random() < 0.4:  # 40% chance to return to (0, 0)
                        next_point = (0, 0)
                    else:
                        radius = random.uniform(self.radius_lower, self.radius_upper)
                        theta = random.uniform(0, 2 * math.pi)
                        next_point = (radius * math.cos(theta), radius * math.sin(theta))
                    switch_count = 1
                x = current_point[0] + (next_point[0] - current_point[0]) * switch_count / self.switch_dur
                y = current_point[1] + (next_point[1] - current_point[1]) * switch_count / self.switch_dur
                coordinate = (x, y)
                switch_count += 1
                if switch_count > self.switch_dur:
                    gaze_interval = random.randint(25, 45)
                    switch_count = 0
            # the first and second rig stands for eye gaze
            pred[i][0] = coordinate[0]
            pred[i][1] = coordinate[1]
        return pred


