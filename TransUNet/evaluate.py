# evaluate_transunet.py
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.data import DataLoader
import loader
import os
import sys
import argparse
import logging
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import kornia.losses

# 配置参数
BATCH_SIZE = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, config):
    """加载预训练的TransUNet模型"""
    net = ViT_seg(config, img_size=256).to(DEVICE)
    
    state_dict = torch.load(model_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    net.eval()
    return net

from skimage.metrics import structural_similarity as ssim
import numpy as np
import torch

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

def save_results(metrics, model_path):
    """保存评估结果到parameters.txt"""
    results = "\n\nTransUNet Evaluation Results:\n"
    results += f"Model: {model_path}\n"
    for name, value in metrics.items():
        results += f"{name}: {value:.6f}\n"
    
    with open("parameters.txt", 'a') as f:
        f.write(results)
    print("✅ Evaluation results saved to parameters.txt")

if __name__ == "__main__":
    # 配置参数 (与原始test.py保持一致)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Radio', help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int, default=2, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()

    # 1. 加载模型
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    model_path ="/home/haohan/Mywork/Code/mytransu/TransUNet/model/Radio256/pretrain_R50-ViT-B_16_skip2_500_epo100_bs12_lr0.0001_256/best_model.pth"
    model = load_model(model_path, config_vit)
    
    # 2. 准备测试数据集
    Radio_test = loader.RadioUNet_c(phase="test")
    test_loader = DataLoader(Radio_test, batch_size=1, shuffle=False, num_workers=1)
    
    device = torch.device('cuda')
    # 3. 评估模型
    print("Evaluating TransUNet...")
    test_metrics = evaluate_model(model, test_loader,device)
    
    # 4. 打印并保存结果
    print("\nEvaluation Results:")
    for name, value in test_metrics.items():
        print(f"{name}: {value:.6f}")
    
    # save_results(test_metrics, model_path)