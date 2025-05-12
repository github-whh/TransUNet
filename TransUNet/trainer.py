import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from functools import partial
from loader import RadioUNet_c
import torch.nn.functional as F
from utils import calc_loss
def worker_init_fn(worker_id, seed):
    random.seed(seed + worker_id)


def trainer(args, model, snapshot_path):
    train_set = RadioUNet_c(phase="train")
    val_set = RadioUNet_c(phase="val")  # 新增验证集
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    # max_iterations = args.max_iterations
    print("The length of train set is: {}".format(len(train_set)))
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    best_val_loss = float('inf')
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.base_lr,                
    )

    # 定义StepLR调度器（每30个epoch乘以0.1）
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, 
        step_size=20,             
        gamma=0.2                 
    )

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    for epoch_num in iterator:
        logging.info("")
        model.train()
        train_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = calc_loss(outputs, labels, iter_num, epoch_num, max_epoch)  # 使用新的损失函数
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 移除手动LR调整代码，仅记录当前LR
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('info/lr', current_lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            train_loss += loss.item() * inputs.size(0)
            iter_num += 1
            
            if iter_num % 100 == 0:
                print(f"iter_num: {iter_num}, loss: {loss.item():.4f}, lr: {current_lr:.6f}")
        
        train_loss /= len(train_set)
        logging.info(f"Epoch {epoch_num}, Train Loss: {train_loss:.4f}")
        writer.add_scalar('info/train_loss', train_loss, epoch_num)     

        # 验证阶段
        # criterion = torch.nn.MSELoss()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.float().to(device), val_labels.float().to(device)
                val_outputs = model(val_inputs)
                val_loss += calc_loss(val_outputs, val_labels, iter_num, epoch_num, max_epoch) * val_inputs.size(0)
        val_loss /= len(val_set)
        writer.add_scalar('info/val_loss', val_loss, epoch_num)
        logging.info(f"Epoch {epoch_num}, Val Loss: {val_loss:.4f}")

    
        # 每个epoch结束后调用scheduler.step()
        exp_lr_scheduler.step()  # 自动更新学习率
        save_interval = 20  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Best model saved at epoch {epoch_num} with val loss {val_loss:.4f}")
        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break
    end_time = time.time()
    total_time = start_time - end_time

    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"\n\n总训练时间: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"

    # 追加训练时间到参数文件
    with open("parameters.txt", 'a') as f:  # 'a' 表示追加模式
        f.write(time_str)
    print(time)
    writer.close()
    return "Training Finished!"