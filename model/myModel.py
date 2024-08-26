import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_config

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim),
        )
            
    def forward(self, x):
        return self.l(x.squeeze(-1))

class myBackbone(nn.Module):
    def __init__(self, modalities, interface):
        super().__init__()
        self.division = interface.division()
        self.modalities = modalities
        self.encoders = nn.ModuleList(
            [
                LinearEncoder(get_config("default_torchaudio_args")["input_size"], 256)
                for _ in range(len(modalities))
            ]
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * len(modalities), 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, len(self.division)),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = torch.cat([encoder(x[key]) for key, encoder in zip(self.modalities, self.encoders)], dim=-1)
        return self.classifier(encoded)
