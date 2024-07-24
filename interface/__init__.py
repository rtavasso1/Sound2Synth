from .base import BaseInterface, ParameterSpaceLoss
from .dexed import DexedInterface
from .torchsynth import TorchSynthInterface

INTERFACE_MAPPING = {
    "Dexed": DexedInterface,
    "TorchSynth": TorchSynthInterface,
}