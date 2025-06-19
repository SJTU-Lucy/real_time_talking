import io
import pyaudio
import librosa
import numpy as np
import keyboard
import wave
import soundfile as sf
import matplotlib.pyplot as plt


def get_audio(sr=16000, chunk=6400):
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
        print(audio_data.shape)

    stream.stop_stream()
    stream.close()
    p.terminate()


def record_audio(sr=16000, chunk=6400):
    outfilepath = 'test.wav'
    with wave.open(outfilepath, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sr)

        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sr,
                        input=True,
                        frames_per_buffer=chunk)

        while not keyboard.is_pressed('left ctrl'):
            continue

        while keyboard.is_pressed('left ctrl'):
            wf.writeframes(stream.read(chunk))

        stream.stop_stream()
        stream.close()
        p.terminate()


get_audio()