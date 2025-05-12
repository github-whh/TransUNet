import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer
from torchsummary import summary
import sys

# 安装：pip install torchinfo
from torchinfo import summary

def save_model_summary(model, input_size=(2, 2, 256, 256), mode='w'):
    with open('parameters.txt', mode) as f:
        results = summary(model, 
                         input_size=input_size,
                         verbose=0,
                         col_names=["input_size", "output_size", "num_params", "kernel_size"],
                         depth=10)
        
        print("...")
        f.write(str(results))



parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='Radio', help='experiment_name')
parser.add_argument('--max_iterations', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=2, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    args.is_pretrain = True
    args.exp = dataset_name + str(args.img_size)
    snapshot_path = "./model/{}/".format(args.exp)
    snapshot_path = snapshot_path + 'pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    #########
    # torch.cuda.set_device(1)  # 强制后续所有CUDA操作使用cuda
    device = torch.device("cuda")  # 显式指定设备 
    net = ViT_seg(config_vit, img_size=args.img_size).to(device)
    save_model_summary(net, input_size=(2, 2, 256, 256), mode='w')
    #########
    net.load_from(weights=np.load("/home/haohan/Mywork/Code/mytransu/model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz"))
    trainer = {'Radio': trainer}
    trainer["Radio"](args, net, snapshot_path)

