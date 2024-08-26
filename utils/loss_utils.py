import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataset import chains
from torchsynth.config import SynthConfig

import laion_clap
from transformers.debug_utils import DebugUnderflowOverflow
# from interface.torchsynth import key

# from interface.torchsynth import REG_NCLASS


class DistributionalCrossEntropyDistance(nn.Module):
    def forward(self, pred, true):
        return -torch.sum(
            true * torch.log(torch.clip(F.softmax(pred, dim=-1), 1e-9, 1)), dim=-1
        ) / (np.log(true.shape[-1]))


def gaussian_kernel(M, std):
    n = -(M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-(n**2) / sig2)
    return w


# class SmoothedCrossEntropyDistance(nn.Module):
#     def __init__(self, k=5, sigma=0.0):
#         super().__init__()
#         self.k = k
#         self.sigma = sigma
#         self.cache = {}

#     def kernel(self, n):
#         if n not in self.cache:
#             self.cache[n] = torch.exp(
#                 -((torch.arange(-self.k, self.k + 1) / n / self.sigma) ** 2) / 2
#             )
#             self.cache[n] /= self.cache[n].sum()
#         return self.cache[n]

#     def weight(self, true):
#         weights = F.conv1d(
#             true.unsqueeze(1),
#             self.kernel(true.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(true),
#             padding="same",
#         )
#         return F.normalize(weights.squeeze(1), p=1, dim=-1)

#     def forward(self, pred, true):
#         return -torch.sum(
#             self.weight(true).detach()
#             * torch.log(torch.clip(F.softmax(pred, dim=-1), 1e-9, 1)),
#             dim=-1,
#         ) / np.log(true.shape[-1])


class SmoothedCrossEntropyDistance(nn.Module):
    def __init__(self, k=5, sigma=1e-6, eps=1e-9):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        self.k = k
        self.sigma = sigma
        self.eps = eps
        self.cache = {}

    def kernel(self, n):
        if n <= 1:
            raise ValueError(f"Input dimension must be greater than 1, got {n}")

        if n not in self.cache:
            x = torch.arange(-self.k, self.k + 1) / (n * self.sigma)
            self.cache[n] = torch.exp(-(x**2) / 2)
            self.cache[n] /= self.cache[n].sum() + self.eps
        return self.cache[n]

    def weight(self, true):
        if torch.any(torch.isnan(true)) or torch.any(torch.isinf(true)):
            raise ValueError("Input tensor 'true' contains NaN or Inf values")

        weights = F.conv1d(
            true.unsqueeze(1),
            self.kernel(true.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(true),
            padding="same",
        )
        return F.normalize(weights.squeeze(1), p=1, dim=-1)

    def forward(self, pred, true):
        if pred.shape != true.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, true {true.shape}")

        if torch.any(torch.isnan(pred)) or torch.any(torch.isinf(pred)):
            raise ValueError("Input tensor 'pred' contains NaN or Inf values")

        if true.shape[-1] <= 1:
            raise ValueError(f"Last dimension must be > 1, got {true.shape[-1]}")

        softmax_pred = F.softmax(pred, dim=-1)
        log_softmax_pred = torch.log(torch.clamp(softmax_pred, min=self.eps, max=1.0))

        weights = self.weight(true).detach()
        loss = -torch.sum(weights * log_softmax_pred, dim=-1)

        normalizer = np.log(true.shape[-1])
        if normalizer == 0:
            raise ValueError("Normalizer (log of last dimension) is zero")

        return loss / normalizer

    @staticmethod
    def check_backward_hook(grad):
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            raise RuntimeError("NaN or Inf detected in gradient")


class ClassificationLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__()
        self.p = p

    def forward(self, pred, true):
        n = true.shape[-1]
        Z = (n - 1) * (1 / n) ** self.p + (1 - 1 / n) ** self.p
        return torch.sum((F.softmax(pred, dim=-1) - true).abs() ** self.p, dim=-1) / Z


class ClassificationAccuracy(nn.Module):
    def forward(self, pred, true):
        return torch.eq(pred.argmax(dim=-1), true.argmax(dim=-1)).float()


class RegressionLpDistance(nn.Module):
    def __init__(self, p=2.0):
        super().__init__()
        self.p = p

    def forward(self, pred, true):
        return (pred.argmax(dim=-1) - true.argmax(dim=-1)).abs() ** self.p / (
            true.shape[-1] - 1
        )

def softargmax1d(input, beta=100):
    *_, n = input.shape
    input = nn.functional.softmax(beta * input, dim=-1)
    indices = torch.linspace(0, 1, n, device=input.device, dtype=input.dtype)
    result = torch.sum((n - 1) * input * indices, dim=-1)
    return result


def differentiable_sample(input, temperature=1.0):
    # Ensure the input is a logit (pre-softmax) tensor
    *_, n = input.shape

    # Sample from Gumbel distribution
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(input) + 1e-20) + 1e-20)

    # Add Gumbel noise to input logits and divide by temperature
    gumbel_softmax = (input + gumbel_noise) / temperature

    # Apply softmax to get a probability distribution
    probs = F.softmax(gumbel_softmax, dim=-1)

    # Generate indices for the output
    indices = torch.linspace(0, 1, n, device=input.device, dtype=input.dtype)

    # Calculate the weighted sum of the indices based on the probabilities
    result = torch.sum((n - 1) * probs * indices, dim=-1)
    return result


class AudioLoss(nn.Module):
    def __init__(self, scales, synth, device="cuda"):
        super().__init__()
        self.scales = scales
        self.device = device
        synth_class = getattr(chains, synth)
        self.synth = synth_class(
            SynthConfig(
                batch_size=32,
                sample_rate=48000,
                reproducible=False,
                buffer_size_seconds=4,
                no_grad=False,
            )
        ).to(device)
        self.synth.on_post_move_to_device()
        print(self.synth.device)
        self.unnormalizer = {}
        for n, p in self.synth.named_parameters():
            n = n.split(".")
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

    def forward(self, pred, true, from_params=True, orig_len=None):
        # print(pred.shape, true.shape)
        if from_params==True:
            waveform, pred_params = self.params_to_audio(
                pred, self.unnormalizer, logits=True, from_0to1=True
            )#.to(pred.device)
            reference_audio, true_params = self.params_to_audio(
                true, self.unnormalizer, logits=True, from_0to1=True
            )#.to(pred.device)
        elif from_params == 'pass':
            waveform = pred
            reference_audio = true
        else:
            waveform, pred_params = self.params_to_audio(
                pred, self.unnormalizer, logits=True, from_0to1=True
            )#.to(pred.device)
            reference_audio = true
            # print(waveform.shape, reference_audio.shape)

        # print(torch.min(waveform), torch.max(waveform), torch.isnan(waveform).any(), torch.min(reference_audio), torch.max(reference_audio), torch.isnan(reference_audio).any())

        pred_stfts = self.multiscale_fft(
            waveform.squeeze(), self.scales, 0.75
        ) # [2048, 1024, 512, 256, 128, 64]
        true_stfts = self.multiscale_fft(reference_audio.squeeze(), self.scales, 0.75)

        # print(torch.max(pred_stfts[0]), torch.max(true_stfts[0]), torch.min(pred_stfts[0]), torch.min(true_stfts[0]), torch.isnan(pred_stfts[0]).any(), torch.isnan(true_stfts[0]).any())

        pred_stfts_log = [self.safe_log(stft) for stft in pred_stfts]
        true_stfts_log = [self.safe_log(stft) for stft in true_stfts]

        log_scale = 1 # / 2  # if self.stft_loss else 1
        stft_scale = 1 # 116 / 2 # if self.stft_loss else 0 # 116 is scale for e6, and /2 averages terms
        losses = [
            stft_scale * F.l1_loss(pred, true)
            + log_scale * F.l1_loss(pred_log, true_log)
            for pred, true, pred_log, true_log in zip(
                pred_stfts, true_stfts, pred_stfts_log, true_stfts_log
            )
        ]
        loss = sum(losses)
        return loss

    def params_to_audio(self, outputs, unnormalizer, logits, from_0to1):
        n_params = len(unnormalizer)

        if logits:
            outputs = torch.stack(outputs.squeeze(0).chunk(n_params, dim=-1))
            # outputs = differentiable_sample(outputs) / (outputs.shape[-1] - 1)
            outputs = torch.sigmoid(torch.mean(outputs, dim=-1))
            # print(outputs.shape)
        mydict = {}
        assert len(unnormalizer) == len(
            outputs
        ), f"Length mismatch: {len(unnormalizer)} vs {len(outputs)}"
        for (k, f), v in zip(unnormalizer.items(), outputs):
            # if "keyboard" in k:
            #     continue  # Skip keyboard parameters as they are initialized frozen
            mydict[k] = (
                f.from_0to1(v).to(self.device) if from_0to1 else v.to(self.device)
            )
            # if torch.isnan(mydict[k]).any():
            #     print(f"NaN in {k}")
            #     print(v)
        # self.synth.set_parameters(mydict)
        audio = self.synth.output(mydict)
        return audio, mydict

class CLAPLoss(nn.Module):
    def __init__(self, synth, loss_type="l1", device="cuda"):
        super().__init__()
        self.device = device
        self.loss_type = loss_type
        self.model = laion_clap.CLAP_Module(enable_fusion=False)
        self.model.load_ckpt(verbose=False)  # download the default pretrained checkpoint.
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

        self.loss_fn = {
            "l1": nn.L1Loss(),
            "l2": nn.MSELoss(),
            "cosim": nn.CosineSimilarity(dim=-1, eps=1e-6),
        }[loss_type]

        synth_class = getattr(chains, synth)
        self.synth = synth_class(
            SynthConfig(
                batch_size=32,
                sample_rate=48000,
                reproducible=False,
                buffer_size_seconds=4,
                no_grad=False,
            )
        ).to(device)
        self.synth.on_post_move_to_device()
        print(f"{synth} synthesizer initialized on {self.synth.device}")
        self.unnormalizer = {}
        for n, p in self.synth.named_parameters():
            n = n.split(".")
            n = (n[0], n[-1])
            self.unnormalizer[n] = p.parameter_range

    def forward(self, pred, true, from_params=True, orig_len=None):
        if from_params == True:
            waveform, pred_params = self.params_to_audio(
                pred, self.unnormalizer, logits=True, from_0to1=True
            )  # .to(pred.device)
            reference_audio, true_params = self.params_to_audio(
                true, self.unnormalizer, logits=True, from_0to1=True
            )  # .to(pred.device)
        elif from_params == "pass":
            waveform = pred
            reference_audio = true
        else:
            waveform, pred_params = self.params_to_audio(
                pred, self.unnormalizer, logits=True, from_0to1=True
            )  # .to(pred.device)
            reference_audio = true
        pred_embed = self.model.get_audio_embedding_from_data(x=waveform, use_tensor=True)
        true_embed = self.model.get_audio_embedding_from_data(x=reference_audio, use_tensor=True)

        loss = self.loss_fn(pred_embed, true_embed)

        if self.loss_type == 'cosim':
            loss = 1 - loss.mean() # to maximize cosine similarity, minimize 1 - cosim

        return loss        

    def params_to_audio(self, outputs, unnormalizer, logits, from_0to1):
        n_params = len(unnormalizer)
        # print(outputs.shape, n_params)
        if logits:
            outputs = torch.stack(outputs.squeeze(0).chunk(n_params, dim=-1))
            outputs = softargmax1d(outputs) / (outputs.shape[-1] - 1)
        # print(outputs)
        mydict = {}
        assert len(unnormalizer) == len(
            outputs
        ), f"Length mismatch: {len(unnormalizer)} vs {len(outputs)}"
        for (k, f), v in zip(unnormalizer.items(), outputs):
            if "keyboard" in k:
                continue  # Skip keyboard parameters as they are initialized frozen
            mydict[k] = (
                f.from_0to1(v).to(self.device) if from_0to1 else v.to(self.device)
            )
            # unsqueeze first dim for batch dim if not present
            if len(mydict[k].shape) == 0:
                mydict[k] = mydict[k].unsqueeze(0)
            # if torch.isnan(mydict[k]).any():
            #     print(f"NaN in {k}")
            #     print(v)
        # self.synth.set_parameters(mydict)
        # print(mydict)
        audio = self.synth.output(mydict)
        return audio, mydict

audio_loss = AudioLoss([2048, 1024, 512, 256, 128, 64], "Voice")
# _ = DebugUnderflowOverflow(audio_loss)
# torch.autograd.set_detect_anomaly(True)

LOSSES_MAPPING = {
    "audioloss": {
        "regl": audio_loss, # AudioLoss([2048, 1024, 512, 256, 128, 64], "Voice"), # CLAPLoss("Voice", "cosim"), # 
        "clsl": None,
    },
    "maeloss": {
        "regl": ClassificationLpDistance(p=1.0),
        "clsl": ClassificationLpDistance(p=1.0),
    },
    "mseloss": {
        "regl": ClassificationLpDistance(p=2.0),
        "clsl": ClassificationLpDistance(p=2.0),
    },
    "celoss": {
        "regl": SmoothedCrossEntropyDistance(sigma=0.02),
        "clsl": DistributionalCrossEntropyDistance(),
    },
    "regceloss": {
        "regl": SmoothedCrossEntropyDistance(sigma=0.02),
        "clsl": None,
    },
    "clsceloss": {
        "regl": None,
        "clsl": DistributionalCrossEntropyDistance(),
    },
    "mixloss": {
        "regl": RegressionLpDistance(p=1.0),
        "clsl": DistributionalCrossEntropyDistance(),
    },
    "clsacc": {
        "regl": None,
        "clsl": ClassificationAccuracy(),
    },
    "regacc": {
        "regl": ClassificationAccuracy(),
        "clsl": None,
    },
    "regmae": {
        "regl": RegressionLpDistance(p=1.0),
        "clsl": None,
    },
}

# __all__ = ['LOSSES_MAPPING']
