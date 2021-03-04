import sys
import os
from datetime import datetime

import numpy as np
import torch
import json
from utils import audio, text
from utils import build_model
from params.params import Params as hp
from modules.tacotron2 import Tacotron
from scipy.io import wavfile
import requests
from hifi_gan.models import Generator
from hifi_gan.env import AttrDict
from time import time
from ZaG2P.api import G2S, load_model
"""

******************************************************** INSTRUCTIONS ********************************************************
*                                                                                                                            *
*   The script expects input utterances on stdin, every example on a separate line.                                          *
*                                                                                                                            *
*   Different models expect different lines, some have to specify speaker, language, etc.:                                   *
*   ID is used as name of the output file.                                                                                   *
*   Speaker and language IDs have to be the same as in parameters (see hp.languages and hp.speakers).                        *
*                                                                                                                            *
*   MONO-lingual and SINGLE-speaker:    id|single input utterance per line                                                   *
*   OTHERWISE                           id|single input utterance|speaker|language                                           *
*   OTHERWISE with PER-CHARACTER lang:  id|single input utterance|speaker|l1-(length of l1),l2-(length of l2),l1             *
*                                           where the last language takes all remaining character                            *
*                                           exmaple: "01|guten tag jean-paul.|speaker|de-10,fr-9,de"                         *
*   OTHERWISE with accent control:      id|single input utterance|speaker|l1-(len1),l2*0.75:l3*0.25-(len2),l1                *
*                                           accent can be controlled by weighting per-language characters                    *
*                                           language codes must be separated by : and weights are assigned using '*number'   *
*                                           example: "01|guten tag jean-paul.|speaker|de-10,fr*0.75:de*0.25-9,de"            *
*                                           the numbers do not have to sum up to one because they are normalized later       *
*                                                                                                                            *
******************************************************************************************************************************

"""

def get_hifiGAN(filepath):
    config_file = os.path.join('hifi_gan/config_v1.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    print("Loading hifi-gan: '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=torch.device("cuda"))

    generator = Generator(h).to(torch.device("cuda"))
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def from_float(_input, dtype):
    if dtype == np.float64:
        return _input, np.float64
    elif dtype == np.float32:
        return _input.astype(np.float32)
    elif dtype == np.uint8:
        return ((_input * 128) + 128).astype(np.uint8)
    elif dtype == np.int16:
        return (_input * 32768).astype(np.int16)
    elif dtype == np.int32:
        return (_input * 2147483648).astype(np.int32)
    raise ValueError('Unsupported wave file format'.format(_input.dtype))

def hifiGAN_infer(mel, generator):
    with torch.no_grad():
        wav = generator(mel) 
        audio = wav.squeeze()
        audio = audio.cpu().numpy()
    return audio

def split_long_sentence(text, max_words):
    result = []
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        tokens = []
        for word in sub_sen.split():
            tokens.append(word)
            if len(tokens) % max_words == 0:
                tokens.append(",")
        result.append(' '.join(tokens))
    text = ','.join(result)
    result = []
    sen = ""
    for sub_sen in text.strip().split(','):
        sub_sen = sub_sen.strip()
        if len((sen + " " + sub_sen).split()) > max_words:
            result.append(sen)
            sen = ""
        if len(sen) > 0:
            sen += " , "
        sen += sub_sen
    if len(sen) > 0:
        result.append(sen)
    return result

def split_text(text, max_word):
    sens_out = []
    sen_out = ''
    for sen in text.split('.'):
        sen = sen.strip()
        if sen:
            sen = sen + ' . '
            if max_word > len(sen.split()):
                if len(sen_out.split()) < max_word - len(sen.split()):
                    sen_out += sen
                else:
                    sens_out.append(sen_out[:-1])
                    sen_out = sen
            else:
                sens_out.append(sen_out[:-1])
                sen_out = ''
                sens_out.append(sen[:-1])
    sens_out.append(sen_out[:-1])
    return sens_out

def synthesize(model, input_data, force_cpu=False):

    item = input_data.split('|')
    clean_text = item[1]

    if not hp.use_punctuation: 
        clean_text = text.remove_punctuation(clean_text)
    if not hp.case_sensitive: 
        clean_text = clean_text.lower()

    t = torch.LongTensor(text.to_sequence(clean_text, use_phonemes=hp.use_phonemes))

    if hp.multi_language:     
        l_tokens = item[3].split(',')
        t_length = len(clean_text) + 1
        l = []
        for token in l_tokens:
            l_d = token.split('#')
 
            language = [0] * hp.language_number
            for l_cw in l_d[0].split(':'):
                l_cw_s = l_cw.split('*')
                language[hp.languages.index(l_cw_s[0])] = 1 if len(l_cw_s) == 1 else float(l_cw_s[1])

            language_length = (int(l_d[1]) if len(l_d) == 2 else t_length)
            l += [language] * language_length
            t_length -= language_length     
        l = torch.FloatTensor([l])
    else:
        l = None

    s = torch.LongTensor([hp.unique_speakers.index(item[2])]) if hp.multi_speaker else None

    if torch.cuda.is_available() and not force_cpu: 
        t = t.cuda(non_blocking=True)
        if l is not None: l = l.cuda(non_blocking=True)
        if s is not None: s = s.cuda(non_blocking=True)

    s = model.inference(t, speaker=s, language=l).cpu().detach().numpy()
    # s = audio.denormalize_spectrogram(s, not hp.predict_linear)
    return s

full_dict = []
def norm(word):
    if word  not in full_dict:
        r = requests.get(f"http:/localhost:5002/norm/{word}")
        return r.content
    return word

if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Model checkpoint.", default="2.0_loss-149-0.459")
    parser.add_argument("--output", type=str, default="result", help="Path to output directory.")
    parser.add_argument("--cpu", action='store_true', help="Force to run on CPU.")
    parser.add_argument("--save_spec", action='store_true', help="Saves also spectrograms if set.")
    parser.add_argument("--ignore_wav", action='store_true', help="Does not save waveforms if set.")
    parser.add_argument("--vocoder", type=str, default="g_00250000")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--name", type=str, default="sample")
    args = parser.parse_args()

    model = build_model(args.checkpoint, force_cpu=False)
    model.eval()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    f = open("lexicon.txt","r")
    for line in f.readlines():
        full_dict.append(line.strip())
    f.close()

    hifiGAN = get_hifiGAN(args.vocoder)
    sentence = ""
    if args.source is not None:
        f = open(args.source, "r")
        for line in f.readlines():
            sentence += line
        f.close()  
    else:
        sentence = "xin chào các bạn ạ , bali moon . cảm ơn bạn đã lắng nghe #"
    
    sens = split_text(sentence.lower(), 50)
    audio_out = []
    total_time_decode = 0
    
    with torch.no_grad():
        for sen in sens:
            for sub_sen in split_long_sentence(sen, 50):
                sub_sen = sub_sen.strip().strip(',').strip() 
                if sub_sen[-1] != ".":
                    sub_sen += " ,"
                print("Text: "+sub_sen)
                final_input = args.name+"|"+sub_sen+"|1|vi" # 1 is vietnamese speaker, can change between 0,1 ; vi | en-us
                t = time()
                mel = synthesize(model, final_input, force_cpu=False)   

                mel = torch.from_numpy(mel).to(torch.device("cuda"))
                mel = torch.unsqueeze(mel, 0)
                wav = hifiGAN_infer(mel, hifiGAN)
                total_time_decode += time() - t
                audio_out += wav.tolist() #+ [0] * int(0 * 22050)
            audio_out += [0] * int(0.1 * 22050)

    audio_out = np.array(audio_out)
    audio_out = from_float(audio_out, np.float32)
    wavfile.write(args.output+"/"+args.name+".wav", 22050, audio_out)
    print("Total time decode: "+ str(total_time_decode))
