import torch
import numpy as np
from utils import *
import librosa as lbrs
import os
import sys
from dataset.generate_torchsynth_dataset import parse_args
from pyheaven import *

def audio_to_spec(audio, sample_rate, augment=None):
    spec = audio_utils.AudioToSpec(audio).transpose(-1, -2)
    spec = torch.clip(torch.log(spec + 1e-5) / 12., -1, 1)
    if augment is not None:
        spec = spec + torch.randn_like(spec) * augment
        spec = torch.clip(spec, -1, 1)
    return spec

def audio_to_mel(audio, sample_rate, augment=None):
    mel = audio_utils.AudioToMel(audio, sample_rate=sample_rate).transpose(-1, -2)
    mel = torch.clip(torch.log(mel + 1e-5) / 12., -1, 1)
    if augment is not None:
        mel = mel + torch.randn_like(mel) * augment
        mel = torch.clip(mel, -1, 1)
    return mel

def audio_to_mfcc(audio, sample_rate, augment=None):
    mfcc = audio_utils.AudioToMFCC(audio, sample_rate=sample_rate).transpose(-1, -2)
    mfcc = torch.clip(mfcc / 1000., -1, 1)
    if augment is not None:
        mfcc = mfcc + torch.randn_like(mfcc) * augment
        mfcc = torch.clip(mfcc, -1, 1)
    return mfcc

def audio_to_features(file, augment=None, save=False):
    audio, sample_rate, params = torch.load(file).values()
    
    label = (params, "")
    # print(audio, sample_rate, label)
    spec = audio_to_spec(audio, sample_rate, augment)
    mel = audio_to_mel(audio, sample_rate, augment)
    mfcc = audio_to_mfcc(audio, sample_rate, augment)

    features = MemberDict({})
    features['label'] = label
    features['sample_rate'] = sample_rate
    features['main'] = spec
    features['mel'] = mel
    features['mfcc'] = mfcc

    features['spec_amp'] = spec.max(dim=-1, keepdim=True)[0]
    features['mel_amp'] = mel.max(dim=-1, keepdim=True)[0]

    features['spec_rms'] = (spec**2).mean(dim=-1, keepdim=True)**0.5
    features['mel_rms'] = (mel**2).mean(dim=-1, keepdim=True)**0.5

    basic_args = {
        'hop_length': get_config("default_torchaudio_args")["hop_length"],
    }
    features['zcr'] = torch.tensor(lbrs.feature.zero_crossing_rate(y=audio.numpy(), **basic_args)).transpose(-1, -2).float()
    features['rms'] = torch.tensor(lbrs.feature.rms(y=audio.numpy(), **basic_args)).transpose(-1, -2).float()

    fft_args = {
        'n_fft': get_config("default_torchaudio_args")["n_fft"],
        'win_length': get_config("default_torchaudio_args")["win_length"],
        'hop_length': get_config("default_torchaudio_args")["hop_length"],
    }
    features['flatness'] = torch.tensor(lbrs.feature.spectral_flatness(y=audio.numpy(), **fft_args)).transpose(-1, -2).float()

    fmins = [32.703, 65.406, 130.81, 261.63, 523.25, 1046.5, 2093.0, 4186.0, 8372.0]
    oct_cqt = [
        torch.tensor(lbrs.feature.chroma_cqt(y=audio.numpy(), sr=sample_rate, n_octaves=1, fmin=fmins[oct - 1], n_chroma=48, bins_per_octave=48)).transpose(-1, -2).float() for oct in range(1, 10)
    ]
    features['chroma'] = torch.cat(oct_cqt, dim=-1)
    
    if save:
        # save in different folder
        path = file.split('/')
        path[-2] = 'processed'
        path = '/'.join(path)
        torch.save(features, path)

    return features

if __name__ == '__main__':
    args_config = {
        '--chain': 'SimpleSynth',
        '--name': 'SimpleSynth',
    }
    args = parse_args(sys.argv[1:], args_config)
    name = args['--name']
    
    for split in ['train', 'test']:
        path = f'data/{name}/{split}/unprocessed/'

        files = os.listdir(path)
        if len(files) == 0:
            print(f"No files found in {path}")
            
        # create folder if not exists
        os.makedirs('/'.join(path.split('/')[:-2])+'/processed', exist_ok=True)
        
        for i, file in enumerate(files):
            if file.endswith('.pt'):
                features = audio_to_features(path + file, save=True)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(files)} {split} files")