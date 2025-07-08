# -*- coding: utf-8 -*-
# !/usr/bin/env python36

from __future__ import print_function

import wave
import threading
import pyaudio


class UnknownValueError(Exception):
    pass


class MyCustomWarning(UserWarning):
    pass



class AudioPlay:
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    def play_wav_file(self, filename, chunk=1024):
        wf = wave.open(filename, 'rb')

        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        stream.start_stream()
        chunk = 16000
        data = wf.readframes(chunk)

        while data != b'':
            stream.write(data)
            data = wf.readframes(chunk)

        stream.stop_stream()
        stream.close()
        wf.close()
        p.terminate()

    def play_audio_data(self, audio_data):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        output=True)
        stream.start_stream()
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def play_wav_file_thread(self, *args, **kwargs):
        thread = threading.Thread(target=self.play_wav_file,
                                  args=args,
                                  kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread


    def play_audio_data_thread(self, *args, **kwargs):
        thread = threading.Thread(target=self.play_audio_data,
                                  args=args,
                                  kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread




class AudioRecognition(object):
    def __init__(self,CHUNK = 1024,FORMAT = pyaudio.paInt16,CHANNELS = 1,RATE = 16000):
        self.CHUNK = CHUNK
        self.FORMAT = FORMAT
        self.CHANNELS = CHANNELS
        self.RATE = RATE

        p = pyaudio.PyAudio()
        self.stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    def recording(self):
        return  self.stream.read(self.CHUNK)

    def record(self,RECORD_SECONDS):
        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            frames.append(data)
        return frames

    def save_data_to_file(self,frames, wav_file):
        wf = wave.open(wav_file, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2) # p.get_sample_size(FORMAT)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

