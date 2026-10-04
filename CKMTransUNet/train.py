"""Train the original May 2025 CKMTransUNet architecture and loss."""
import argparse
import copy
from pathlib import Path
import random

import numpy as np
import torch

from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pretrained", required=True, help="R50+ViT-B_16.npz initialization")
    parser.add_argument("--output_dir", default="model/Radio256")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("Training requires CUDA; evaluation also supports CPU")
    if args.n_gpu < 1 or args.n_gpu > torch.cuda.device_count():
        parser.error("--n_gpu must not exceed the available CUDA device count")
    if not Path(args.pretrained).is_file():
        parser.error(f"Pretrained initialization not found: {args.pretrained}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = copy.deepcopy(CONFIGS_ViT_seg["R50-ViT-B_16"])
    config.n_skip = 2
    config.patches.grid = (16, 16)
    model = ViT_seg(config, img_size=256).cuda()
    model.load_from(weights=np.load(args.pretrained))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    from trainer import trainer
    trainer(args, model, args.output_dir)


if __name__ == "__main__":
    main()
