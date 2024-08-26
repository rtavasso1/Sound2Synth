import warnings

# Filter out all user warnings
warnings.filterwarnings("ignore", category=UserWarning)

import typing
import torch
import sys
import os
import json
from pathlib import Path
from torchsynth.config import SynthConfig, BASE_REPRODUCIBLE_BATCH_SIZE
try:
    from . import chains
except:
    import chains

def parse_args(args: str, args_config: dict) -> dict:
    if len(args) % 2 != 0:
        raise Exception(f'Odd number of arguments. Check them!!!\nHere are the available arguments: {args_config}')

    dic = args_config.copy()
    i = 1
    while i < len(args):
        arg_key = args[i-1]
        arg_val = args[i]

        if arg_key not in args:
            raise Exception(f'Arg {arg_key} is unknown. Here are the available arguments: {args_config}')

        dic[arg_key] = type(args_config[arg_key])(arg_val)
        i += 2
        
    return dic

def main():
    args_config = {
        '--chain': 'SimpleSynth',
        '--name': 'SimpleSynth',
        '--num_batches': 100,
        '--split': 0.8,
        '--dir': 'data',
        '--batch_size': 1024,
        '--sample_rate': 48000,
        '--duration_seconds': 4,
        '--dry_run': False,
    }

    args = parse_args(sys.argv[1:], args_config)
    print(f'Processing with the following arguments: {args}', flush=True)

    base_dir = Path(args['--dir']) / args['--name']
    train_dir = base_dir / 'train' / 'unprocessed'
    test_dir = base_dir / 'test' / 'unprocessed'
    
    if not args['--dry_run']:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        with open(base_dir / 'arguments.json', 'w') as f:
            json.dump(args, f, indent=4)

    synthconfig = SynthConfig(
        batch_size=args['--batch_size'],
        sample_rate=args['--sample_rate'],
        buffer_size_seconds=args['--duration_seconds'],
    )
    synth = getattr(chains, args['--chain'])(synthconfig)
    
    if torch.cuda.is_available():
        synth.to('cuda')

    n = 0
    split = args['--split']
    for i in range(args['--num_batches']):
        out = synth(i)
        is_training = (i / args['--num_batches']) < split

        for audio, params in zip(out[0], out[1]):
            dic = {
                'audio': audio.unsqueeze(0).detach().cpu(),
                'sample_rate': args['--sample_rate'],
                'params': params.detach().cpu(),
            }
            # print(params.shape)
            parent_dir = train_dir if is_training else test_dir
            if not args['--dry_run']:
                torch.save(dic, parent_dir / f'sound_{n}.pt')
            n += 1

        print(f"Generating batch {i+1} in {Path(parent_dir).parent.name}", flush=True)

if __name__ == '__main__':
    main()
