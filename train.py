import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from utils import *
from dataset import DATASET_MAPPING, DATASET_PATHS
from interface import INTERFACE_MAPPING
from model import get_backbone, get_classifier, Net

import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.callbacks import (
    StochasticWeightAveraging,
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from typing import Any, Dict
import traceback
import inspect

import wandb
wandb.require("core")

from sound2synth import Sound2SynthModel, SplitDatasets, Identifier

from transformers.debug_utils import DebugUnderflowOverflow

if __name__ == "__main__":
    args = HeavenArguments.from_parser(
        [
            LiteralArgumentDescriptor(
                "synth",
                short="s",
                choices=INTERFACE_MAPPING.keys(),
                default="TorchSynth",
            ),
            LiteralArgumentDescriptor(
                "dataset_type",
                short="dt",
                choices=DATASET_MAPPING.keys(),
                default="my_multimodal",
            ),
            StrArgumentDescriptor("dataset", short="ds", default="SimpleSynth"),
            StrArgumentDescriptor("chain", short="ch", default="SimpleSynth"),
            StrArgumentDescriptor("project", short="pj", default="SynthGPT"),
            StrArgumentDescriptor("run_name", short="rn", default="Base"),
            StrArgumentDescriptor("backbone", short="bb", default="multimodal"),
            SwitchArgumentDescriptor("multimodal_use_spec", short="use_spec"),
            StrArgumentDescriptor("classifier", short="cl", default="parameter"),
            IntArgumentDescriptor("feature_dim", short="f", default=2048),
            IntArgumentDescriptor("num_epochs", short="e", default=50),
            IntArgumentDescriptor("batch_size", short="b", default=32),
            IntArgumentDescriptor("grad_accum", short="ga", default=-64),
            IntArgumentDescriptor("examples", short="ex", default=128),
            FloatArgumentDescriptor("limit_train_batches", short="limtr", default=1.0),
            FloatArgumentDescriptor("limit_val_batches", short="limvl", default=1.0),
            FloatArgumentDescriptor("learning_rate", short="lr", default=2e-4),
            FloatArgumentDescriptor("warmup_start_lr_ratio", short="wlr", default=0.01),
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
            SwitchArgumentDescriptor("wandb", short="wandb"),
            SwitchArgumentDescriptor("audio_loss", short="audio_loss"),
            # Add a new argument for the checkpoint path
            StrArgumentDescriptor("resume_from_checkpoint", short="cp", default=None),
        ]
    )
    if args.clean:
        CMD("rm -rf lightning_logs/*")
        CMD("rm -rf examples/*")
        CMD("rm -rf logs/*")
    if args.grad_accum < 0:
        args.grad_accum = max(-args.grad_accum // args.batch_size, 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.n_gpu = len([d for d in args.cuda.split(",") if d.strip() != ""])

    torch.set_float32_matmul_precision("medium")

    seed_everything(args.seed, workers=True)

    # Prepare interface
    interface = INTERFACE_MAPPING[args.synth]

    # Prepare network
    net = Net(
        backbone=get_backbone(args.backbone, args),
        classifier=get_classifier(args.classifier, interface, args),
    )

    # Save Arguments
    args.identifier = Identifier(args)
    SaveJson(args, pjoin("tasks", f"{args.identifier}.json"), indent=4)

    data_dir = lambda dataset: "data/" + dataset if dataset != "Vengeance" else dataset + "/processed"

    # Prepare dataset
    args.datasets = MemberDict(
        {
            "train": DATASET_MAPPING[args.dataset_type](
                dir=data_dir(args.dataset), chain=args.chain, split="train"
            ),
            "val": DATASET_MAPPING[args.dataset_type](
                dir=data_dir(args.dataset), chain=args.chain, split="test"
            ),
            "test": DATASET_MAPPING[args.dataset_type](
                dir=data_dir(args.dataset), chain=args.chain, split="test"
            ),
        }
    )
    # print(len(args.datasets["train"].files), args.dataset, data_dir(args.dataset))
    # Prepare W&B logger
    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.project,
            name=args.dataset + "_" + args.run_name,
        )

    # Train model
    model = Sound2SynthModel(net, interface, args=args)
    # debug = DebugUnderflowOverflow(model)

    class EnhancedLog2Detector(pl.Callback):
        def __init__(self):
            super().__init__()
            self.log2_detected = False
            self.log2_locations = []

        def _inspect_graph(self, tensor: torch.Tensor, name: str):
            if tensor.grad_fn is None:
                return

            def traverse(fn, depth=0):
                if 'Div' in type(fn).__name__:
                    self.log2_detected = True
                    location_info = self._get_location_info()
                    self.log2_locations.append(f"{name} -> {location_info}")

                if hasattr(fn, 'next_functions'):
                    for next_fn, _ in fn.next_functions:
                        if next_fn is not None:
                            traverse(next_fn, depth + 1)

            traverse(tensor.grad_fn)

        def _get_location_info(self):
            stack = inspect.stack()
            for frame_info in stack:
                if frame_info.function == 'forward' or frame_info.function == 'calculate_loss':
                    filename = os.path.basename(frame_info.filename)
                    return f"In {filename}, line {frame_info.lineno}, in {frame_info.function}"
            return "Location not found"

        def on_after_backward(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            self.log2_detected = False
            self.log2_locations = []

            # Inspect model parameters
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    self._inspect_graph(param.grad, name)

            # Inspect the loss value
            if hasattr(pl_module, 'last_loss'):
                self._inspect_graph(pl_module.last_loss, 'loss')

            if self.log2_detected:
                print("Log2 operation detected in computational graph")
                for location in self.log2_locations:
                    print(f"Log2 found at: {location}")

                pl_module.log('log2_detected', float(self.log2_detected), on_step=True, on_epoch=True)

    # Assuming `model` is your neural network
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         param.register_hook(lambda grad, name=name: detect_log2_grad_hook(grad, name))

    class GradientMonitorCallback(pl.Callback):
        def on_after_backward(self, trainer, pl_module):
            # Loop through all model parameters
            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad)
                    if torch.isnan(grad_norm):
                        print(f"NaN gradient in {name}")
                    elif grad_norm > 1e3:  # Adjust this threshold as needed
                        print(f"High gradient norm detected in {name}: {grad_norm}")

    trainer = pl.Trainer(
        detect_anomaly=True,
        max_epochs=(
            args.num_epochs
            if args.resume_from_checkpoint is None
            else args.num_epochs + 10
        ),
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=args.grad_accum,
        callbacks=[
            # StochasticWeightAveraging(swa_lrs=0.05),
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                monitor="valid_audioloss",
                dirpath=f"checkpoints/{args.run_name}/",
                filename="ckpt-{epoch:02d}-{valid_celoss:.2f}",
                save_top_k=1,
                mode="min",
                save_last=True,
            ),
            # EnhancedLog2Detector(),
            # GradientMonitorCallback(),
            # EarlyStopping(
            #     monitor="valid_celoss", mode="min", patience=8, check_finite=True
            # ),
        ],
        # GPU configuration
        accelerator="gpu",
        devices=args.n_gpu,
        # Logging configuration
        logger=wandb_logger if args.wandb else True,
        log_every_n_steps=100,
        # Speedup configuration
        # benchmark=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        precision=32,
        # resume_from_checkpoint=(
        #     args.resume_from_checkpoint if args.resume_from_checkpoint else None
        # ),
    )

    if args.wandb:
        wandb_logger.watch(model)
    trainer.fit(model)
    # trainer.test(ckpt_path="best")
