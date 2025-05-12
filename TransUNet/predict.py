import numpy as np
from PIL import Image
import torch
import os
import csv
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from skimage import io

total = 0
class TransUNetPredictor:
    def __init__(self, model_path, img_size=256, vit_name='R50-ViT-B_16', n_skip=2, vit_patches_size=16):
        self.img_size = img_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model configuration
        config_vit = CONFIGS_ViT_seg[vit_name]
        config_vit.n_skip = n_skip
        config_vit.patches.size = (vit_patches_size, vit_patches_size)
        if vit_name.find('R50') != -1:
            config_vit.patches.grid = (int(img_size/vit_patches_size), int(img_size/vit_patches_size))
        
        # Create model and load weights
        self.model = ViT_seg(config_vit, img_size=img_size).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
    
    def predict(self, x, y, building_path, antenna_path):
        global total
        import time
        """Make prediction for a single coordinate pair"""
        # Load input images
        input_buildings = io.imread(building_path) / 255.0
        input_Tx = io.imread(antenna_path) / 255.0
        
        # Prepare input tensor
        inputs = np.stack([input_buildings, input_Tx], axis=2)
        inputs = np.transpose(inputs, (2, 0, 1))
        inputs = torch.from_numpy(inputs).unsqueeze(0).float().to(self.device)
        # Model prediction
        time1 = time.time()
        with torch.no_grad():
            pred1 = self.model(inputs)
            time2 = time.time()
            total += time2 - time1
        
        builds = inputs.detach().cpu().numpy()[0, 0]
        indB = builds != 0
        
        # Create output directory for this pair
        pair_dir = f"/home/haohan/Mywork/Code/png/{x}_{y}/"
        os.makedirs(pair_dir, exist_ok=True)
        
        # Process each beam
        for beam_id in range(8):
            beam_pred = (256 * pred1[0][beam_id].detach().cpu().numpy()).astype(np.uint8)
            
            im = np.zeros([self.img_size, self.img_size, 3], dtype=np.uint8)
            im[:, :, 0] = beam_pred  # R
            im[:, :, 1] = beam_pred  # G
            im[:, :, 2] = beam_pred  # B
            im[indB, 2] = 255  # Buildings in blue
            
            # Save with TransUNet naming convention
            im_pred = Image.fromarray(im)
            pred_filename = f"{pair_dir}{x}_{y}_transUNet{beam_id}.png"
            im_pred.save(pred_filename)
        
        print(f"Processed pair ({x}, {y}) - saved to {pair_dir}")
        return pred1

def read_coordinates_from_csv(csv_path):
    """Read x,y coordinates from CSV file"""
    coordinates = []
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x = int(row['x'])
            y = int(row['y'])
            coordinates.append((x, y))
    return coordinates

# Usage example
if __name__ == "__main__":
    # Configuration parameters
    MODEL_PATH = "/home/haohan/Mywork/Code/mytransu/TransUNet/model/Radio256/pretrain_R50-ViT-B_16_skip2_500_epo100_bs12_lr0.0001_256/best_model.pth"
    CSV_PATH = "/home/haohan/Mywork/Code/coordinates.csv"
    
    # Create predictor
    predictor = TransUNetPredictor(model_path=MODEL_PATH)
    
    # Read coordinates from CSV
    coordinates = read_coordinates_from_csv(CSV_PATH)
    
    # Process each coordinate pair
    for x, y in coordinates:
        building_path = f"/home/haohan/Mywork/Code/RadioMapSeer/png/buildings_complete/{x}.png"
        antenna_path = f"/home/haohan/Mywork/Code/RadioMapSeer/png/antennas/{x}_{y}.png"
        predictor.predict(x, y, building_path, antenna_path)
    
    print("All predictions completed!")
    print(total)