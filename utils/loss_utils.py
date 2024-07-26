import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataset import chains
from torchsynth.config import SynthConfig
# from interface.torchsynth import REG_NCLASS

class DistributionalCrossEntropyDistance(nn.Module):
    def forward(self, pred, true):
        return -torch.sum(true*torch.log(torch.clip(F.softmax(pred,dim=-1), 1e-9, 1)), dim=-1) / (np.log(true.shape[-1]))

def gaussian_kernel(M, std):
    n =  - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

class SmoothedCrossEntropyDistance(nn.Module):
    def __init__(self, k=5, sigma=0.0):
        super().__init__(); self.k = k; self.sigma = sigma; self.cache = {}
    def kernel(self, n):
        if n not in self.cache:
            self.cache[n] = torch.exp(-(torch.arange(-self.k, self.k+1)/n/self.sigma)**2/2); self.cache[n] /= self.cache[n].sum()
        return self.cache[n]
    def weight(self, true):
        weights = F.conv1d(true.unsqueeze(1),self.kernel(true.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(true),padding='same'); return F.normalize(weights.squeeze(1), p=1, dim=-1)
    def forward(self, pred, true):
        return -torch.sum(self.weight(true).detach()*torch.log(torch.clip(F.softmax(pred,dim=-1), 1e-9, 1)), dim=-1) / np.log(true.shape[-1])

class ClassificationLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = p
    def forward(self, pred, true):
        n = true.shape[-1]; Z = (n-1)*(1/n)**self.p+(1-1/n)**self.p
        return torch.sum((F.softmax(pred,dim=-1)-true).abs()**self.p, dim=-1) / Z

class ClassificationAccuracy(nn.Module):
    def forward(self, pred, true):
        return torch.eq(pred.argmax(dim=-1),true.argmax(dim=-1)).float()

class RegressionLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__(); self.p = p
    def forward(self, pred, true):
        return (pred.argmax(dim=-1)-true.argmax(dim=-1)).abs()**self.p / (true.shape[-1]-1)
    
class AudioLoss(nn.Module):
    def __init__(self, scales, synth):
        super().__init__()
        self.scales = scales
        synth_class = getattr(chains, synth)
        self.synth = synth_class(SynthConfig(batch_size=32, sample_rate=48000, reproducible=False, buffer_size_seconds=4, no_grad=False))
        self.unnormalizer = {}
        for n, p in self.synth.named_parameters():
            n = n.split('.')
            n = (n[0], n[-1])
            self.unnormalizer[n] = p.parameter_range
        
    def safe_log(self, x):
        return torch.log(x + 1e-6)

    def fft(self, signal, scale, overlap=None, hop_length=None):
        assert overlap or hop_length, "Either overlap or hop_length must be provided"
        if overlap:
            hop_length = int(scale * (1 - overlap))
        S = torch.stft(
            signal,
            n_fft=scale,
            hop_length=hop_length,
            win_length=scale,
            window=torch.hann_window(scale).to(signal),
            center=True,
            normalized=True,
            return_complex=True,
        ).abs()
        return S

    def multiscale_fft(self, signal, scales, overlap=None, hop_length=None):
        assert overlap or hop_length, "Either overlap or hop_length must be provided"
        stfts = []
        for s in scales:
            S = self.fft(signal, s, overlap, hop_length)
            stfts.append(S)
        return stfts
    
    def forward(self, pred, true):
        waveform = self.params_to_audio(pred, self.unnormalizer, logits=True, from_0to1=True)
        reference_audio = self.params_to_audio(true, self.unnormalizer, logits=True, from_0to1=(true.max() <= 1).item())
        
        pred_stfts = self.multiscale_fft(waveform.squeeze(), self.scales, 0.75) # [2048, 1024, 512, 256, 128, 64]
        true_stfts = self.multiscale_fft(reference_audio.squeeze(), self.scales, 0.75)

        pred_stfts_log = [self.safe_log(stft) for stft in pred_stfts]
        true_stfts_log = [self.safe_log(stft) for stft in true_stfts]

        log_scale = 1/2 # if self.stft_loss else 1
        stft_scale = 116/2 # if self.stft_loss else 0 # 116 is scale for e6, and /2 averages terms
        losses = [stft_scale*F.l1_loss(pred, true) + log_scale*F.l1_loss(pred_log, true_log) for pred, true, pred_log, true_log in zip(pred_stfts, true_stfts, pred_stfts_log, true_stfts_log)]
        loss = sum(losses)
        return loss
    
    def params_to_audio(self, outputs, unnormalizer, logits, from_0to1):
        n_params = len(unnormalizer)
 
        if logits:
            outputs = torch.stack(outputs.squeeze(0).chunk(n_params, dim=-1))
            outputs = outputs.argmax(dim=-1)/(outputs.shape[-1]-1)

        mydict = {}
        assert len(unnormalizer) == len(outputs), f"Length mismatch: {len(unnormalizer)} vs {len(outputs)}"
        for (k, f), v in zip(unnormalizer.items(), outputs):
            if 'keyboard' in k:
                continue # Skip keyboard parameters as they are initialized frozen
            mydict[k] = f.from_0to1(v) if from_0to1 else v
            # print(mydict[k].shape)
            
        self.synth.set_parameters(mydict)
        # print(self.synth.keyboard.torchparameters.midi_f0)
        audio = self.synth.output()
        return audio
        

LOSSES_MAPPING = {
    'audioloss': {
        'regl': AudioLoss([2048, 1024, 512, 256, 128, 64], 'Synplant2'),
        'clsl': None,
    },
    'maeloss': {
        'regl': ClassificationLpDistance(p=1.0),
        'clsl': ClassificationLpDistance(p=1.0),
    },
    'mseloss': {
        'regl': ClassificationLpDistance(p=2.0),
        'clsl': ClassificationLpDistance(p=2.0),
    },
    'celoss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'regceloss': {
        'regl': SmoothedCrossEntropyDistance(sigma=0.02),
        'clsl': None,
    },
    'clsceloss': {
        'regl': None,
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'mixloss': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': DistributionalCrossEntropyDistance(),
    },
    'clsacc': {
        'regl': None,
        'clsl': ClassificationAccuracy(),
    },
    'regacc': {
        'regl': ClassificationAccuracy(),
        'clsl': None,
    },
    'regmae': {
        'regl': RegressionLpDistance(p=1.0),
        'clsl': None,
    }
}

# __all__ = ['LOSSES_MAPPING']