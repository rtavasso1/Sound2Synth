import utils
import utils.metrics as metrics
from utils.audio_utils import *
import torch
import os
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchsynth.config import SynthConfig
from dataset import chains
import torchsynth
import torchaudio

# suppress torch warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def find_all_files(directory):
    file_paths = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Join the root directory and file name to get the full file path
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


class MyMultiModalDataset(IterableDataset):
    def __init__(self, dir, chain, split, shard_size=100):
        self.dir = os.path.join(dir, split, 'processed')
        self.files = find_all_files(self.dir)
        self.shard_size = shard_size
        self.num_shards = (len(self.files) + shard_size - 1) // shard_size  # Calculate total number of shards

        # self.unnormalizer = {}
        # synth_class = getattr(chains, chain) # SimpleSynth, Synplant2, Voice
        # self.synth = synth_class(SynthConfig(batch_size=1, sample_rate=48000, reproducible=False, buffer_size_seconds=4))
        # for n, p in self.synth.named_parameters():
        #     n = n.split('.')
        #     n = (n[0], n[-1])
        #     self.unnormalizer[n] = p.parameter_range

    def _load_shard(self, shard_index):
        start = shard_index * self.shard_size
        end = min(start + self.shard_size, len(self.files))
        shard_files = self.files[start:end]
        features = [torch.load(file) for file in shard_files]
        return features

    def _worker_init_fn(self):
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading, return the full iterator
            self.worker_id = 0
            self.num_workers = 1
        else:  # In a worker process
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        self.current_shard_index = self.worker_id
        self.features = []
        self.local_index = 0
        self._load_next_shard()

    def _load_next_shard(self):
        if self.current_shard_index < self.num_shards:
            self.features = self._load_shard(self.current_shard_index)
            self.current_shard_index += self.num_workers
            self.local_index = 0
        else:
            self.features = []

    def __iter__(self):
        self._worker_init_fn()
        return self

    def __next__(self):
        if self.local_index >= len(self.features):
            self._load_next_shard()
            if not self.features:  # No more shards left
                raise StopIteration

        features = self.features[self.local_index]
        self.local_index += 1

        params, grad = features.pop('label')  # tuple of (params, "") where "" is the unspecified gradient
        params = torch.tensor(params).float().detach() if not isinstance(params, torch.Tensor) else params.float().detach()

        # for i, (_, f) in enumerate(self.unnormalizer.items()):
        #     params[i] = f.from_0to1(params[i]).detach()

        label = (params, grad)
        sample_rate = features.pop('sample_rate')

        # TODO: modify data processing to remove the need for this
        features.pop('__dict__')  # whoopsie processing error!

        # TODO: modify data processing to remove the need for this
        for key in features:
            features[key] = torch.tensor(features[key]) if isinstance(features[key], torchsynth.signal.Signal) else features[key]
            features[key] = features[key].squeeze(0).detach()

        return (features, sample_rate), label


class VengeanceDataset(IterableDataset):
    def __init__(self, dir, chain, split, shard_size=100):
        self.files = find_all_files(os.path.join(dir, split))
        self.shard_size = shard_size
        self.num_shards = (
            len(self.files) + shard_size - 1
        ) // shard_size  # Calculate total number of shards

    def _load_shard(self, shard_index):
        start = shard_index * self.shard_size
        end = min(start + self.shard_size, len(self.files))
        shard_files = self.files[start:end]
        try:
            features = [torch.load(file) for file in shard_files]
        except Exception as e:
            print(e)
            self._load_next_shard()
        for i in range(len(shard_files)):
            f = shard_files[i]
            f = f.split("/")
            # remove 2nd and 3rd elements
            f = f[:1] + f[3:]
            f = "/".join(f)
            f = f[:-3] + ".wav"
            a, sr = torchaudio.load(f)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                a = resampler(a)
            a = a.mean(0)[:48000*4]
            orig_len = torch.tensor(a.size(0))
            a = torch.concat((a, torch.zeros(48000*4-a.size(0))), 0)

            features[i]['audio'] = a
            features[i]['orig_len'] = orig_len
        return features

    def _worker_init_fn(self):
        worker_info = get_worker_info()
        if worker_info is None:  # Single-process data loading, return the full iterator
            self.worker_id = 0
            self.num_workers = 1
        else:  # In a worker process
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
        self.current_shard_index = self.worker_id
        self.features = []
        self.local_index = 0
        self._load_next_shard()

    def _load_next_shard(self):
        if self.current_shard_index < self.num_shards:
            self.features = self._load_shard(self.current_shard_index)
            self.current_shard_index += self.num_workers
            self.local_index = 0
        else:
            self.features = []

    def __iter__(self):
        self._worker_init_fn()
        return self

    def __next__(self):
        if self.local_index >= len(self.features):
            self._load_next_shard()
            if not self.features:  # No more shards left
                raise StopIteration

        features = self.features[self.local_index]
        self.local_index += 1

        params, grad = features.pop(
            "label"
        )  # tuple of (params, "") where "" is the unspecified gradient
        params = (
            torch.tensor(params).float().detach()
            if not isinstance(params, torch.Tensor)
            else params.float().detach()
        )
        params = torch.rand(76) # params.unsqueeze(0).repeat(74)

        label = (params, grad)
        sample_rate = features.pop("sample_rate")
        # sample_rate = [('adsr_1', 'attack'), ('adsr_1', 'decay'),('adsr_1', 'sustain'),('adsr_1', 'release'),('adsr_1', 'alpha'),('adsr_2', 'attack'),('adsr_2', 'decay'),('adsr_2', 'sustain'),('adsr_2', 'release'),('adsr_2', 'alpha'),('lfo_1', 'frequency'),('lfo_1', 'mod_depth'),('lfo_1', 'initial_phase'),('lfo_1', 'sin'),('lfo_1', 'tri'),('lfo_1', 'saw'),('lfo_1', 'rsaw'),('lfo_1', 'sqr'),('lfo_2', 'frequency'),('lfo_2', 'mod_depth'),('lfo_2', 'initial_phase'),('lfo_2', 'sin'),('lfo_2', 'tri'),('lfo_2', 'saw'),('lfo_2', 'rsaw'),('lfo_2', 'sqr'),('lfo_1_amp_adsr', 'attack'),('lfo_1_amp_adsr', 'decay'),('lfo_1_amp_adsr', 'sustain'),('lfo_1_amp_adsr', 'release'),('lfo_1_amp_adsr', 'alpha'),('lfo_2_amp_adsr', 'attack'),('lfo_2_amp_adsr', 'decay'),('lfo_2_amp_adsr', 'sustain'),('lfo_2_amp_adsr', 'release'),('lfo_2_amp_adsr', 'alpha'),('lfo_1_rate_adsr', 'attack'),('lfo_1_rate_adsr', 'decay'),('lfo_1_rate_adsr', 'sustain'),('lfo_1_rate_adsr', 'release'),('lfo_1_rate_adsr', 'alpha'),('lfo_2_rate_adsr', 'attack'),('lfo_2_rate_adsr', 'decay'),('lfo_2_rate_adsr', 'sustain'),('lfo_2_rate_adsr', 'release'),('lfo_2_rate_adsr', 'alpha'),('mod_matrix', 'adsr_1->vco_1_pitch'),('mod_matrix', 'adsr_1->vco_1_amp'),('mod_matrix', 'adsr_1->vco_2_pitch'),('mod_matrix', 'adsr_1->vco_2_amp'),('mod_matrix', 'adsr_1->noise_amp'),('mod_matrix', 'adsr_2->vco_1_pitch'),('mod_matrix', 'adsr_2->vco_1_amp'),('mod_matrix', 'adsr_2->vco_2_pitch'),('mod_matrix', 'adsr_2->vco_2_amp'),('mod_matrix', 'adsr_2->noise_amp'),('mod_matrix', 'lfo_1->vco_1_pitch'),('mod_matrix', 'lfo_1->vco_1_amp'),('mod_matrix', 'lfo_1->vco_2_pitch'),('mod_matrix', 'lfo_1->vco_2_amp'),('mod_matrix', 'lfo_1->noise_amp'),('mod_matrix', 'lfo_2->vco_1_pitch'),('mod_matrix', 'lfo_2->vco_1_amp'),('mod_matrix', 'lfo_2->vco_2_pitch'),('mod_matrix', 'lfo_2->vco_2_amp'),('mod_matrix', 'lfo_2->noise_amp'),('vco_1', 'mod_depth'),('vco_1', 'initial_phase'),('vco_2', 'mod_depth'),('vco_2', 'initial_phase'),('vco_2', 'shape'),('mixer', 'vco_1'),('mixer', 'vco_2'), ('mixer', 'noise')]
        sample_rate = 48000

        # TODO: modify data processing to remove the need for this
        features.pop("__dict__")  # whoopsie processing error!

        # TODO: modify data processing to remove the need for this
        for key in features:
            features[key] = (
                torch.tensor(features[key])
                if isinstance(features[key], torchsynth.signal.Signal)
                else features[key]
            )
            features[key] = features[key].squeeze(0).detach()
            
        # audio normalization
        # max_val = 2. # torch.max(torch.abs(features['audio']))
        # features['audio'] = features['audio'] #/ max_val
        # features['audio'] = torch.clamp(features['audio'], -1, 1)

        return (features, sample_rate), label
