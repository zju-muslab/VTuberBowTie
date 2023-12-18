import time
import numpy as np
import torch
import StreamVC
from pydub import AudioSegment
from pydub.playback import play
import torchaudio
import matplotlib.pyplot as plt
import yaml
from munch import Munch
from warnings import simplefilter
import pyaudio

# Ignore warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

# Set basic parameters
config_path = 'config.yml'

# Load parameters from config file
with open(config_path) as f:
    parameters = yaml.safe_load(f)
    parameters = Munch(parameters)

RATE = 24000
FORMAT = pyaudio.paFloat32
CHANNELS = 1
pa = pyaudio.PyAudio()

# Load start and end beeps
start_beep = AudioSegment.from_wav("./misc/music/start.wav")
end_beep = AudioSegment.from_wav("./misc/music/end.wav")

# Initialize StreamVC object
stvc = StreamVC.StreamVC(parameters)

# Initialize buffers for received audio, output audio, and input audio
receive_buff, outputs_wav, inputs_wav = [], [], []

# Flag to indicate if conversion is enabled
conversion_flag = False


def play_callback(in_data, frame_count, time_info, status):
    """
    Callback function for audio playback.
    """
    global receive_buff
    start = time.time()
    cnt = 0
    while len(receive_buff) == 0:
        cnt += 1
        if cnt > 100:
            return (None, pyaudio.paContinue)
        time.sleep(0.01)
    while len(receive_buff) > 1:
        receive_buff.pop(0)
    x, delay_start = receive_buff.pop(0)  # Allow maximum delay of 1 CHUNK
    y = stvc.feed(x)
    if conversion_flag:
        data = y.cpu().numpy().astype(np.float32)
    else:
        data = x.astype(np.float32)
    outputs_wav.append(data)
    end = time.time()
    # print(
    #     f'{end-start:.3f}/{end-delay_start:.3f}/{parameters["CHUNK"]/24000:.3f}s, rtf={(end-start)/(parameters["CHUNK"]/24000):.3f}')
    return (data.tobytes(), pyaudio.paContinue)


def record_callback(in_data, frame_count, time_info, status):
    """
    Callback function for audio recording.
    """
    global receive_buff
    receive = np.frombuffer(in_data, dtype=np.float32)
    receive_buff.append([receive, time.time()])
    inputs_wav.append(receive)
    return None, pyaudio.paContinue


if __name__ == '__main__':
    play(start_beep)

    print('Welcome to using VTuerBowTie!')
    print('- Press "t + Enter" to turn on/off voice changer')
    print('- Press "1-9 + Enter" to switch voices')
    print('- Press "c + Enter" to exit')

    # Open audio streams for recording and playback
    record_stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                            frames_per_buffer=parameters['CHUNK'], stream_callback=record_callback)
    play_stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True,
                          frames_per_buffer=parameters['CHUNK'], stream_callback=play_callback)

    time.sleep(0.2)
    record_stream.start_stream()
    play_stream.start_stream()

    conversion_flag = False

    '''running commands
    '''
    while True:
        command = input()
        if command in ['t', 'T']:  # Toggle conversion on/off
            conversion_flag = not conversion_flag
        elif command in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:  # Switch voice based on number
            # TODO: Implement voice switching logic
            pass
        else:
            break

    # Stop and close audio streams
    record_stream.stop_stream()
    record_stream.close()
    while play_stream.is_active():
        time.sleep(0.1)
    play_stream.stop_stream()
    play_stream.close()
    pa.terminate()

    time.sleep(0.2)
    play(end_beep)
    print('Thank you very much!')

    # Save output and input audio as WAV files
    wav = np.concatenate(outputs_wav, axis=0).reshape(1, -1)
    torchaudio.save('./output/output.wav', torch.Tensor(wav), 24000)
    wav = np.concatenate(inputs_wav, axis=0).reshape(1, -1)
    torchaudio.save('./output/input.wav', torch.Tensor(wav), 24000)
