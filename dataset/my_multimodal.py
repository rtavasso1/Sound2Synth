import utils
import utils.metrics as metrics
from utils.audio_utils import *
import torch
import os
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torchsynth.config import SynthConfig
from dataset import chains

class MyMultiModalDataset(IterableDataset):
    def __init__(self, dir, chain, split, shard_size=100):
        self.dir = os.path.join(dir, split, 'processed')
        self.files = os.listdir(self.dir)
        self.shard_size = shard_size
        self.num_shards = (len(self.files) + shard_size - 1) // shard_size  # Calculate total number of shards
        
        self.unnormalizer = {}
        synth_class = getattr(chains, chain) # SimpleSynth, Synplant2, Voice
        self.synth = synth_class(SynthConfig(batch_size=1, sample_rate=48000, reproducible=False, buffer_size_seconds=4))
        for n, p in self.synth.named_parameters():
            n = n.split('.')
            n = (n[0], n[-1])
            self.unnormalizer[n] = p.parameter_range

    def _load_shard(self, shard_index):
        start = shard_index * self.shard_size
        end = min(start + self.shard_size, len(self.files))
        shard_files = self.files[start:end]
        features = [torch.load(os.path.join(self.dir, file)) for file in shard_files]
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
            features[key] = features[key].squeeze(0)

        return (features, sample_rate), label