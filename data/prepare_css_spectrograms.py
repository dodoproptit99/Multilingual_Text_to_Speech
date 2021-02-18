import sys
import os
import numpy as np

sys.path.insert(0, "../")

from utils import audio
from params.params import Params as hp
from audio import audio as audio_processing

if __name__ == '__main__':
    import argparse
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument("--ljspeech", type=str, default="ljspeech")
    parser.add_argument("--numienbac", type=str, default="numienbac")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate.")
    parser.add_argument("--num_fft", type=int, default=1024, help="Number of FFT frequencies.")
    args = parser.parse_args()

    hp.sample_rate = args.sample_rate
    hp.num_fft = args.num_fft

    files_to_solve = [
        (args.numienbac, "train.txt"),
        (args.numienbac, "val.txt"),
        # (args.ljspeech, "train.txt"),
        # (args.ljspeech, "val.txt"),
    ]

    spectrogram_dirs = [#os.path.join(args.ljspeech, 'spectrograms'), 
                        #os.path.join(args.ljspeech, 'linear_spectrograms'),
                        os.path.join(args.numienbac, 'spectrograms'), 
                        os.path.join(args.numienbac, 'linear_spectrograms')]
    for x in spectrogram_dirs:
        if not os.path.exists(x): os.makedirs(x)

    metadata = []
    for d, fs in files_to_solve:
        with open(os.path.join(d,fs), 'r', encoding='utf-8') as f:
            metadata.append((d, fs, [line.rstrip().split('|') for line in f]))

    print(f'Please wait, this may take a very long time.')
    for d, fs, m in metadata:  
        print(f'Creating spectrograms for: {fs}')

        with open(os.path.join(d, fs), 'w', encoding='utf-8') as f:
            for i in m:
                idx, s, l, a, raw_text = i
                spec_name = idx + '.npy'      
                audio_path = os.path.join(d, a)       
                audio_data = audio.load(audio_path)

                splitted_a = a.split("/")
                if splitted_a[0] == "..":
                    mel_path_partial = os.path.join(splitted_a[0], splitted_a[1], "spectrograms", spec_name)
                    lin_path_partial = os.path.join(splitted_a[0], splitted_a[1], "linear_spectrograms", spec_name)
                else:
                    mel_path_partial = os.path.join("spectrograms", spec_name)
                    lin_path_partial = os.path.join("linear_spectrograms", spec_name)

                mel_path = os.path.join(d, mel_path_partial)
                if not os.path.exists(mel_path):
                    mel, energy = audio_processing.tools.get_mel(audio_path)
                    mel = mel.numpy().astype(np.float32)
                    np.save(mel_path, mel)
                    # np.save(mel_path, audio.spectrogram(audio_data, True))
                lin_path = os.path.join(d, lin_path_partial)
                if not os.path.exists(lin_path):
                    np.save(lin_path, audio.spectrogram(audio_data, False))

                print(f'{idx}|{s}|{l}|{a}|{mel_path_partial}|{lin_path_partial}|{raw_text}', file=f)