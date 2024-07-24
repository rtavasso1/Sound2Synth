import warnings
warnings.filterwarnings("ignore",category=UserWarning)

from utils import *
from dataset import DATASET_MAPPING, DATASET_PATHS
from interface import INTERFACE_MAPPING
from model import get_backbone, get_classifier, Net

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor, ModelCheckpoint, EarlyStopping

from sound2synth import Sound2SynthModel, SplitDatasets, Identifier

if __name__=="__main__":
    args = HeavenArguments.from_parser([
        LiteralArgumentDescriptor("synth", short="s", choices=INTERFACE_MAPPING.keys(), default="TorchSynth"),
        LiteralArgumentDescriptor("dataset_type", short="dt", choices=DATASET_MAPPING.keys(), default="my_multimodal"),
        StrArgumentDescriptor("dataset", short="ds", default="SimpleSynth"),
        StrArgumentDescriptor("project", short="pj", default="SynthGPT"),
        StrArgumentDescriptor("run_name", short="rn", default="Base"),

        StrArgumentDescriptor("backbone", short="bb", default="multimodal"),
        SwitchArgumentDescriptor("multimodal_use_spec", short="use_spec"),
        StrArgumentDescriptor("classifier", short="cl", default="parameter"),
        IntArgumentDescriptor("feature_dim", short="f", default=2048),
        IntArgumentDescriptor("num_epochs", short="e", default=30),
        IntArgumentDescriptor("batch_size", short="b", default=32),
        IntArgumentDescriptor("grad_accum", short="ga", default=-64),
        IntArgumentDescriptor("examples", short="ex", default=128),

        FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
        FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
        FloatArgumentDescriptor("learning_rate", short="lr", default=2e-4),
        FloatArgumentDescriptor("warmup_start_lr_ratio", short="wlr", default=.01),
        FloatArgumentDescriptor("eta_min", short="em", default=1e-8),
        IntArgumentDescriptor("warmup_epochs", short="we", default=4),
        FloatArgumentDescriptor("weight_decay", short="wd", default=1e-4),
        FloatArgumentDescriptor("noise_augment", short="na", default=1e-4),
        FloatArgumentDescriptor("mode_dropout", short="dp", default=0.0),

        StrArgumentDescriptor("identifier", short="id", default=None),
        StrArgumentDescriptor("cuda", short="cd", default="0"),
        IntArgumentDescriptor("seed", short="sd", default=20001101),
        IntArgumentDescriptor("debug", default=-1),
        SwitchArgumentDescriptor("clean"),
        SwitchArgumentDescriptor("wandb",short='wandb'),
    ])
    if args.clean:
        CMD("rm -rf lightning_logs/*")
        CMD("rm -rf examples/*")
        CMD("rm -rf logs/*")
    if args.grad_accum < 0:
        args.grad_accum = -args.grad_accum//args.batch_size
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.n_gpu = len([d for d in args.cuda.split(',') if d.strip()!=''])
    
    torch.set_float32_matmul_precision('medium')

    seed_everything(args.seed, workers=True)
    
    # Prepare interface
    interface = INTERFACE_MAPPING[args.synth]
    
    # Prepare network
    net = Net(
        backbone = get_backbone(args.backbone, args),
        classifier = get_classifier(args.classifier, interface, args),
    )

    # Prepare dataset
    args.datasets = MemberDict({
        'train': DATASET_MAPPING[args.dataset_type](dir=DATASET_PATHS[args.dataset], split='train'),
        'val' : DATASET_MAPPING[args.dataset_type](dir=DATASET_PATHS[args.dataset], split='test'),
        'test' : DATASET_MAPPING[args.dataset_type](dir=DATASET_PATHS[args.dataset], split='test'),
    })
    
    # Train model
    model = Sound2SynthModel(net, interface, args=args)
