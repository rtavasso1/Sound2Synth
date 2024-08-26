from .base import AudioDataset
from .spec import SpecDataset, MelDataset, MFCCDataset
from .multimodal import MultiModalDataset
from .my_multimodal import MyMultiModalDataset, VengeanceDataset

DATASET_MAPPING = {
    'audio': AudioDataset,
    'spec': SpecDataset,
    'mel': MelDataset,
    'mfcc': MFCCDataset,
    'multimodal': MultiModalDataset,
    'my_multimodal': MyMultiModalDataset,
    'vengeance': VengeanceDataset,
}

DATASET_PATHS = {
    'SimpleSynth': 'data/SimpleSynth',
    'SimpleSynth_fixedKeybaord': 'data/SimpleSynth_fixedKeybaord',
    'SimpleSynth_fixedKeybaordBig': 'data/SimpleSynth_fixedKeybaordBig',
    'SimpleSynth_goodFix': 'data/SimpleSynth_goodFix',
    'Synplant2': 'data/Synplant2',
}