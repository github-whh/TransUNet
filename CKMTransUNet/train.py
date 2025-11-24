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

from torchinfo import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def save_model_summary(model, input_size=(2, 2, 256, 256), mode='w'):
    """
    Save model architecture summary to a file
    """
    with open('parameters.txt', mode) as f:
        results = summary(model, input_size=input_size, verbose=0, 
                         col_names=["input_size", "output_size", "num_params", "kernel_size"], 
                         depth=10) 
        print("Generating model summary...")
        f.write(str(results))

# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Debug', help='Experiment name')
parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
parser.add_argument('--n_gpu', type=int, default=2, help='Total number of GPUs')
parser.add_argument('--deterministic', type=int, default=1, help='Whether to use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001, help='Segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=256, help='Input patch size of network input')
parser.add_argument('--seed', type=int, default=1234, help='Random seed')
parser.add_argument('--n_skip', type=int, default=2, help='Number of skip connections to use')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='ViT model selection')
parser.add_argument('--vit_patches_size', type=int, default=16, help='ViT patches size, default is 16')

args = parser.parse_args()

if __name__ == "__main__":
    # Set up deterministic training if specified
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Set up experiment configuration
    dataset_name = args.dataset
    args.is_pretrain = True
    args.exp = dataset_name
    
    # Create snapshot path for model saving
    snapshot_path = "./model/{}/".format(args.exp)
    snapshot_path = snapshot_path + 'pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    # Create directory if it doesn't exist
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    # Initialize Vision Transformer model
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_skip = args.n_skip
    
    # Configure patches grid for R50 models
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), 
                                 int(args.img_size / args.vit_patches_size))

    # Set up device and model
    device = torch.device("cuda")  # Explicitly specify device
    net = ViT_seg(config_vit, img_size=args.img_size).to(device)
    
    # Save model architecture summary
    save_model_summary(net, input_size=(2, 2, 256, 256), mode='w')

    # Load pre-trained weights
    net.load_from(weights=np.load("./pretrained/R50+ViT-B_16.npz"))
    
    # Initialize trainer and start training
    trainer_dict = {'CKM': trainer}
    trainer_dict["CKM"](args, net, snapshot_path)