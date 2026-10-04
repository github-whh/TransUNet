# CKMTransUNet: original checkpoint evaluation

This package uses the verified May 12, 2025 model implementation: `R50-ViT-B_16`,
`n_skip=2`, image size 256, and ResNet max-pool `padding=0`.

## Run

Use Python 3.10. Unzip the package and run inside `CKMTransUNet`:

```bash
python -m pip install -r requirements.txt
python evaluate.py --model /path/to/best_model.pth --data_dir /path/to/BeamCKMSeer
```

[Model download](https://drive.google.com/drive/folders/1bJUFdAlnKic8MdM0r8QDP3cmG-SSOkcq?usp=drive_link)
| [Dataset download](https://drive.google.com/drive/folders/1rXx10-FE3ALH-57TAh9_2ltZEt0JnjDk?usp=drive_link)

The checkpoint and dataset are supplied separately. The dataset root must contain:

```text
png/buildings_complete/1.png
png/antennas/1_0.png
data/1_0/0.png ... 7.png
```

The script uses the original 1,000-sample test split, FP32, and batch size 1.
RMSE and PSNR are computed per sample, then averaged. CUDA is used when available;
add `--device cpu` for CPU or `--num_workers 0` if worker processes are unavailable.

Verified results: MAE 0.019102, NMSE 0.056952, RMSE 0.044545, PSNR 27.349985,
SSIM 0.832431. Small numerical differences may occur between environments.

Verified checkpoint SHA-256:
`57773f33383815b7815aabbdab9607950e04c8ffc2c016c0df85b9a4a971e52c`.

Original model source: [commit 1f0f8d1](https://github.com/github-whh/TransUNet/tree/1f0f8d1506db806085b7ed7d0d97fbaa5b0567fd/TransUNet).
