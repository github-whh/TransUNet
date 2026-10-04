# CKMTransUNet

Code for [Beamforming-Codebook-Aware Channel Knowledge Map Construction for Multi-Antenna Systems](https://arxiv.org/abs/2505.16132).

`CKMTransUNet/` contains the restored May 2025 architecture compatible with the
released `best_model.pth`: `n_skip=2` and ResNet max-pool `padding=0`.
It preserves the original evaluation metrics and test split.

## Evaluate

See the short [evaluation README](CKMTransUNet/README.md) for model/data downloads and verified results.

```bash
cd CKMTransUNet
python -m pip install -r requirements.txt
python evaluate.py --model /path/to/best_model.pth --data_dir /path/to/BeamCKMSeer
```

## Train

The original training loop and loss are included. Supply the ViT initialization
`R50+ViT-B_16.npz` and a complete BeamCKMSeer dataset:

```bash
python -m pip install -r requirements-training.txt
python train.py --data_dir /path/to/BeamCKMSeer --pretrained /path/to/R50+ViT-B_16.npz
```

The default training settings are 100 epochs, batch size 12, learning rate 0.0001,
and one GPU; use `--n_gpu 2` when training with two GPUs.

Based on [TransUNet](https://github.com/Beckschen/TransUNet).
Contact: whh24@mails.tsinghua.edu.cn.
