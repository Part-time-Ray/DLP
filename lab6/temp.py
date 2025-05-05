from torchvision.utils import save_image
from unet import UNet
from utils import BetaScheduler
from dataloader import get_test_dataset
from evaluator import evaluation_model
from torchvision import transforms
import torch
import json
import os
from dataloader import Dataset
from evaluator import evaluation_model
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import BetaScheduler, inference
from unet_2 import UNet



def main(args):
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    # model = UNet(args.max_time_step, in_channels=3, out_channels=3).to(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    beta_scheduler = BetaScheduler(num_diffusion_timesteps=args.max_time_step, beta_start=1e-4, beta_end=0.02, device=device)
    while True:
        label = list(map(int, input('input label: ').split()))
        label = torch.tensor([1 if i in label else 0 for i in range(24)], device=device).unsqueeze(0)
        image = inference(model, beta_scheduler, label, image_size=(3, 64, 64), device=device)
        save_image(image[0], 'temp.png', normalize=True)
        with open('objects.json', 'r', encoding='utf-8') as file:
            objects = json.load(file)
        label = label[0].cpu().numpy()
        for name in objects:
            if label[objects[name]]:
                print(name)

# 6 9 22
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-p', type=str, default='result/best_model_st_1000_unet_2_beta_2_linear.pt', help='Path to the trained model')
    parser.add_argument('--max_time_step', '-st', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='GPU index to use')
    args = parser.parse_args()
    
    main(args)

