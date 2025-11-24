import logging
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from loader import BeamCKM
from functools import partial
from lossFunction import L2_loss

def worker_init_fn(worker_id, seed):
    random.seed(seed + worker_id)

def trainer(args, model, snapshot_path):
    train_set = BeamCKM(phase="train")
    val_set = BeamCKM(phase="val")
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%m/%d %H:%M')
    logging.info(str(args))

    # max_iterations = args.max_iterations
    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=args.seed), pin_memory=True)
    valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, worker_init_fn=partial(worker_init_fn, seed=args.seed), pin_memory=True)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=80)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_val_loss = float('inf')
    start_time = time.time()
    for epoch_num in iterator:
        print("")
        logging.info("")
        model.train()
        train_loss = 0.0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = model(inputs)
            loss = L2_loss(outputs, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('info/lr', current_lr, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            train_loss += loss.item() * inputs.size(0)
            iter_num += 1
            
            if iter_num % 100 == 0:
                print(f"iter_num: {iter_num}, loss: {loss.item():.6f}, lr: {current_lr:.6f}")
        
        train_loss /= len(train_set)
        logging.info(f"Epoch {epoch_num}, Train Loss: {train_loss:.6f}")
        writer.add_scalar('info/train_loss', train_loss, epoch_num)     

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in valloader:
                val_inputs, val_labels = val_inputs.float().to(device), val_labels.float().to(device)
                val_outputs = model(val_inputs)
                val_loss += L2_loss(val_outputs, val_labels) * val_inputs.size(0)
        val_loss /= len(val_set)
        writer.add_scalar('info/val_loss', val_loss, epoch_num)
        logging.info(f"Epoch {epoch_num}, Val Loss: {val_loss:.6f}")

        exp_lr_scheduler.step()
        save_interval = 20
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Best model saved at epoch {epoch_num} with val loss {val_loss:.6f}")
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
    time_str = f"\n\nTraining Time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"

    with open("parameters.txt", 'a') as f:
        f.write(time_str)
    print(time)
    writer.close()
    return "Training Finished!"