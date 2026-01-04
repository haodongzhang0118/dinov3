#!/usr/bin/env python
"""
Inference for DIET finetuning experiments.
"""

import os
import argparse
import shutil
import time
import logging
import yaml

import torch
import wandb
from transformers import AutoModel

from config import create_experiment_config_from_args, DEVICE
from loaders.data_loader import prepare_data_loaders
from utils.wandb_logger import (
    create_experiment_dashboard,
    log_inference_metrics_summary_table,
)
from models.utils import set_reproducibility_seeds, dcp_folder_to_torch_save
from evaluation.metrics import zero_shot_eval
from models.huggingface_to_dinov3 import convert_dinov3_to_huggingface
from models.dinov2 import DINOWrapper

logger = logging.getLogger("test_dinov3")


def get_hf_model_from_checkpoint_folder(checkpoint_folder_path):
    config_path = os.path.join(checkpoint_folder_path, "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert "cp" in cfg and cfg["cp"].get("enabled", False), (
        "'cp' section not found in config.yaml or cp disabled. Are you loading a non-CP checkpoint?"
    )
    cp_hf_model = cfg["cp"]["hf_model"]

    return cp_hf_model


def load_dinovx_from_checkpoint(checkpoint_folder_path, hf_model):
    # we have the following folder structure:
    # checkpoint_folder_path/
    #   ├── ckpt/
    #   │    ├── final/
    #   │    │    ├── SOME_NAME.distcp (DINOv3) OR: SOME_NAME.pth (DINOv2)
    #   │    ├── ... (other checkpoints, irrelevant)
    #   ├── config.yaml
    # Load config.yaml from the base path

    final_dir = os.path.join(checkpoint_folder_path, "ckpt", "final")
    torch_ckpt = dcp_folder_to_torch_save(final_dir)
    state_dict = torch.load(torch_ckpt, map_location="cpu", weights_only=False)
    converted_dict = convert_dinov3_to_huggingface(state_dict, use_student=False)
    hf_model.load_state_dict(converted_dict, strict=False)

    return hf_model


def test(args):
    """Main inference function"""
    print("\n" + "=" * 70)
    print("DIET FINETUNING INFERENCE for DINOvX models")
    print("=" * 70)

    # Basic settings
    device = DEVICE
    print(f"Using device: {device}")

    # Create the backbone model
    cp_hf_model = get_hf_model_from_checkpoint_folder(args.local_dino_checkpoint_folder)
    model = AutoModel.from_pretrained(cp_hf_model)
    model = DINOWrapper(model, version="v3")
    model.to(device)

    # add stuff to args for experiment config creation
    args.backbone = "dinov3_cp"
    args.model_size = cp_hf_model.split("/")[-1].split("-")[1]

    # Load data
    dataset_name = args.dataset

    print(f"\nLoading {dataset_name} dataset...")
    train_loader, val_loader, test_loader, dataset_info = prepare_data_loaders(
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        da_strength=args.da_strength,
        limit_data=args.limit_data,
        root=args.data_root,
    )

    num_classes = dataset_info["num_classes"]
    num_diet_classes = dataset_info["num_diet_classes"]
    print(f"Loaded dataset with {num_classes} classes")
    print(f"Dataset size determines {num_diet_classes} diet classes")

    # Create experiment configuration
    config = create_experiment_config_from_args(args)

    # Convert to wandb format for logging
    experiment_config = config.to_wandb_config()

    run = None

    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_prefix,
            entity=args.wandb_entity,
            dir=args.wandb_dir,
            config=experiment_config,
            settings=wandb.Settings(start_method="thread"),
        )
        print(f"Initialized wandb run: {run.name}")

        # Log that we're starting inference
        run.log({"inference_started": True}, commit=False)

    # initial kNN and LP eval
    print("\n" + "=" * 50)
    print("INITIAL ZERO-SHOT EVALUATION (BEFORE DIET CONTINUED PRETRAINING)")
    print("=" * 50)
    initial_time = time.time()

    initial_results = zero_shot_eval(
        model=model,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
        store_embeddings=args.store_embeddings,
    )
    print(f"Initial evaluation completed in {time.time() - initial_time:.2f}s")

    # Load Continued Pretraining checkpoint
    print("\nLoading Continued Pretraining checkpoint...")

    # model is wrapped in DINOWrapper, so we need to access the underlying model
    model.model = load_dinovx_from_checkpoint(
        args.local_dino_checkpoint_folder, model.model
    )
    initial_time = time.time()

    final_results = zero_shot_eval(
        model=model,
        train_loader=train_loader,
        test_loader=(test_loader if args.eval_on_test else val_loader),
        device=device,
        probe_lr=1e-3,
        probe_steps=10000,
        store_embeddings=args.store_embeddings,
    )

    print(f"Final evaluation completed in {time.time() - initial_time:.2f}s")

    if run is not None:
        log_inference_metrics_summary_table(
            run=run,
            wandb_id=run.id,
            backbone_type=args.backbone,
            model_size=args.model_size,
            dataset=dataset_name,
            initial_metrics=initial_results,
            final_metrics=final_results,
        )

    # Create experiment dashboard in wandb
    if args.use_wandb and run is not None:
        create_experiment_dashboard(
            run, None, initial_results, final_results, experiment_config
        )

        # Log that inference is complete
        run.log({"inference_completed": True})

        # Finish the wandb run
        run.finish()
        print("Inference results logged to the original wandb run")

    return initial_results, final_results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DIET Finetuning Framework")

    # Dataset arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Directory to store dataset files.",
    )
    parser.add_argument(
        "--limit-data",
        type=int,
        default=1000,
        help="Maximum number of training samples for kNN/LP training set",
    )

    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="Number of training epochs",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
        help="Weight decay",
    )
    parser.add_argument(
        "--da-strength",
        type=int,
        default=2,
        help="Data augmentation strength (0-3)",
    )

    # DIET arguments
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.3,
        help="Label smoothing strength (0 to disable DIET)",
    )
    parser.add_argument(
        "--diet-head-only-epochs",
        type=float,
        default=0.05,
        help="Fraction of total epochs for DIET-head-only training (freezes backbone). "
        "Set to 0.0 for direct full training.",
    )
    parser.add_argument(
        "--num-trained-blocks",
        type=int,
        default=-1,
        help="Number of transformer blocks to train from the end of the backbone. "
        "Set to -1 to train all blocks, 0 to freeze all blocks, "
        "4 to train last 4 blocks.",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5,
        help="Run zero-shot evaluation every N epochs",
    )
    parser.add_argument(
        "--eval-on-test",
        action="store_true",
        help="Evaluate on test set instead of validation set",
    )

    # Logging and saving arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-wandb",
        dest="use_wandb",
        action="store_false",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity (organization) name",
    )
    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="Existing wandb run ID to attach inference results to",
    )
    parser.add_argument(
        "--wandb-dir",
        type=str,
        default="wandb",
        help="Directory to save wandb logs",
    )
    parser.add_argument(
        "--wandb-prefix",
        type=str,
        default="DIET_INFERENCE",
        help="Prefix for wandb experiment names",
    )
    parser.add_argument(
        "--run-sanity-check",
        action="store_true",
        help="Run the initial k-NN sanity check on CIFAR10.",
    )
    parser.add_argument(
        "--store-embeddings",
        action="store_true",
        help="Whether to store embeddings from the model",
    )
    parser.add_argument(
        "--local-dino-checkpoint-folder",
        type=str,
        default="",
        help="Path to a local DINO checkpoint folder to use. "
        "NOTE: This folder should contain the checkpoint file, and train config.",
    )

    # parameters are explicitly set here
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name for finetuning (e.g., octmnist)",
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    # Set reproducibility seeds
    set_reproducibility_seeds(args.seed)

    # Create directories if they don't exist
    os.makedirs(args.wandb_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)

    # Run inference
    test(args)


if __name__ == "__main__":
    main()
