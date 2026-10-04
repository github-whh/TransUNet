"""Evaluate the May 2025 CKMTransUNet checkpoint using its original metrics."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

from loader import RadioUNet_c
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg


def load_model(model_path, config, device):
    net = ViT_seg(config, img_size=256)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    net.load_state_dict(state_dict, strict=True)
    return net.to(device).eval()


def evaluate_model(model, dataloader, device):
    model.eval()
    total_mse = 0.0
    total_rmse = 0.0
    total_nrmse = 0.0
    total_mae = 0.0
    total_ssim = 0.0
    total_nmse = 0.0
    total_psnr = 0.0
    valid_ssim_samples = 0  # 记录有效SSIM计算样本数
    count = 0
    
    # 初始化损失函数
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()
            
            # 输入形状应为 (B, C, H, W)
            outputs = model(inputs)
            
            for i in range(outputs.shape[0]):  # 遍历batch中的每个样本
                # 当前样本 (C, H, W)
                output = outputs[i]  # (C, H, W)
                target = targets[i]  # (C, H, W)
                
                # 计算 MSE 和 RMSE
                mse = mse_loss(output, target).item()
                rmse = np.sqrt(mse)
                total_mse += mse
                total_rmse += rmse
                
                # 计算 NRMSE
                data_range = target.max() - target.min()
                nrmse = rmse / (data_range.item() + 1e-10)
                total_nrmse += nrmse
                
                # 计算 NMSE
                nmse = mse / (torch.mean(target**2).item() + 1e-10)
                total_nmse += nmse
                
                # 计算 MAE
                mae = l1_loss(output, target).item()
                total_mae += mae
                
                # 计算 PSNR
                if mse == 0:
                    psnr = 100.0
                else:
                    psnr = 20 * np.log10(data_range.item() / np.sqrt(mse))
                total_psnr += psnr
                
                # 转换到numpy并处理维度 (C, H, W) -> (H, W, C)
                output_np = output.cpu().numpy().transpose(1, 2, 0)
                target_np = target.cpu().numpy().transpose(1, 2, 0)
                
                # 计算多通道SSIM
                ssim_val = 0
                valid_channels = 0
                
                for c in range(output_np.shape[2]):  # 遍历通道
                    channel_target = target_np[..., c]
                    channel_output = output_np[..., c]
                    
                    channel_range = channel_target.max() - channel_target.min()
                    if channel_range < 1e-6:  # 跳过无效通道
                        continue
                        
                    # 计算单通道SSIM
                    current_ssim = ssim(
                        channel_output,
                        channel_target,
                        data_range=channel_range,
                        win_size=11)  # 自适应窗口
                    
                    ssim_val += current_ssim
                    valid_channels += 1
                
                # 只有至少有一个有效通道时才计入SSIM
                if valid_channels > 0:
                    ssim_val /= valid_channels
                    total_ssim += ssim_val
                    valid_ssim_samples += 1
                
                count += 1
    
    # 计算平均值
    metrics = {
        'MSE': total_mse / count,
        'RMSE': total_rmse / count, 
        'NRMSE': total_nrmse / count,
        'NMSE': total_nmse / count,
        'MAE': total_mae / count,
        'SSIM': total_ssim / valid_ssim_samples if valid_ssim_samples > 0 else float('nan'),
        'PSNR': total_psnr / count,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="best_model.pth", help="Path to the released checkpoint")
    parser.add_argument("--data_dir", required=True, help="Dataset root containing png/ and data/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not Path(args.model).is_file():
        parser.error(f"Checkpoint not found: {args.model}")
    if not all((Path(args.data_dir) / name).is_dir() for name in ("png", "data")):
        parser.error("--data_dir must contain png/ and data/ directories")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("--batch_size must be positive and --num_workers must be nonnegative")

    torch.set_num_threads(4)
    device = torch.device(args.device)
    config = copy.deepcopy(CONFIGS_ViT_seg["R50-ViT-B_16"])
    config.n_skip = 2
    config.patches.size = (16, 16)
    config.patches.grid = (16, 16)
    model = load_model(args.model, config, device)
    dataset = RadioUNet_c(phase="test", dir_dataset=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Evaluating {len(dataset)} samples on {device} (n_skip=2, pool padding=0)...", flush=True)
    start = time.time()
    metrics = evaluate_model(model, dataloader, device)
    print("\nEvaluation Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")
    print(f"Elapsed: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()
