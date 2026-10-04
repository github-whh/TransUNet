# CKMTransUNet

CKMTransUNet constructs beamforming-codebook-aware channel knowledge maps for multi-antenna systems.

For more information, see the paper: [Beamforming-Codebook-Aware Channel Knowledge Map Construction for Multi-Antenna Systems](https://arxiv.org/abs/2505.16132).

`CKMTransUNet/` contains the restored May 2025 architecture compatible with the
released `best_model.pth`: `n_skip=2` and ResNet max-pool `padding=0`.
It preserves the original evaluation metrics and test split.

## Model

Download the trained `best_model.pth` from [Google Drive](https://drive.google.com/drive/folders/1bJUFdAlnKic8MdM0r8QDP3cmG-SSOkcq?usp=drive_link).

Use the restored `CKMTransUNet/` code below to evaluate this checkpoint.

## Dataset

Download the **BeamCKMSeer dataset** from [Google Drive](https://drive.google.com/drive/folders/1rXx10-FE3ALH-57TAh9_2ltZEt0JnjDk?usp=drive_link).

Extract the dataset and pass its root directory, containing `png/` and `data/`, as `--data_dir`.

## Evaluate

See the short [evaluation README](CKMTransUNet/README.md) for the dataset layout and verified results.

```bash
cd CKMTransUNet
python -m pip install -r requirements.txt
python evaluate.py --model /path/to/best_model.pth --data_dir /path/to/BeamCKMSeer
```

## Train CKMTransUNet

The original training loop and loss are included. Supply the ViT initialization
`R50+ViT-B_16.npz` and a complete BeamCKMSeer dataset:

```bash
python -m pip install -r requirements-training.txt
python train.py --data_dir /path/to/BeamCKMSeer --pretrained /path/to/R50+ViT-B_16.npz
```

The default training settings are 100 epochs, batch size 12, learning rate 0.0001,
and one GPU; use `--n_gpu 2` when training with two GPUs.

## Acknowledgement

This code is based on the [TransUNet](https://github.com/Beckschen/TransUNet) repository. We thank the authors for their valuable work.

Feel free to contact whh24@mails.tsinghua.edu.cn if you have any problem.
