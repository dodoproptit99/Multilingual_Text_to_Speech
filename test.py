import sys
import os
import numpy as np

from utils import audio
from params.params import Params as hp
from audio import audio as audio_processing

if __name__ == '__main__':
       
    audio_data = audio.load("test.wav")
    mel = audio.spectrogram(audio_data, True)
    print(mel.shape)
    mel, _ = audio_processing.tools.get_mel("test.wav")   
    mel = mel.numpy().astype(np.float32)
    print(mel.shape)