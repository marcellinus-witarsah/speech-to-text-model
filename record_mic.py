import pyaudio
import wave
import numpy as np

p = pyaudio.PyAudio()

# Constants
CHUNK = 3200
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SECONDS = 1


def record_sound():
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    print("start recording ...")
    frames = []
    for i in range(0, int(RATE / CHUNK * SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("stop recording ...")
    stream.stop_stream()
    stream.close()
    return np.frombuffer(b''.join(frames), dtype=np.int16)


def terminate():
    p.terminate()

# with wave.open('output_by_marcel.wav', 'wb') as wf:
#     wf.setnchannels(CHANNELS)
#     wf.setsampwidth(p.get_sample_size(FORMAT))
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()
