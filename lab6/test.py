from torchvision.utils import save_image
# from unet import UNet
# from unet_2 import UNet
from unet_3 import UNet
from utils import BetaScheduler, inference
from dataloader import get_test_dataset
from evaluator import evaluation_model
from torchvision import transforms
import torch
import os

def main(args):
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    # model = UNet(args.max_time_step, in_channels=3, out_channels=3).to(device)
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    beta_scheduler = BetaScheduler(num_diffusion_timesteps=args.max_time_step, beta_start=1e-4, beta_end=0.02, device=device)
    label, new_label = get_test_dataset(device=device)
    image = beta_scheduler.reverse_all(model, label, image_size=(3, 64, 64), mode='test')
    # image = inference(model, beta_scheduler, label, image_size=(3, 64, 64), device=device)
    os.makedirs(os.path.join('images', 'test'), exist_ok=True)
    for i in range(image.shape[0]):
        save_image(image[i], os.path.join('images', 'test', f'{i}.png'), normalize=True)
    print("Saved images to images/test")

    new_image = beta_scheduler.reverse_all(model, new_label, image_size=(3, 64, 64), mode='new_test')
    # new_image = inference(model, beta_scheduler, new_label, image_size=(3, 64, 64), device=device)
    os.makedirs(os.path.join('images', 'new_test'), exist_ok=True)
    for i in range(new_image.shape[0]):
        save_image(new_image[i], os.path.join('images', 'new_test', f'{i}.png'), normalize=True)
    print("Saved new images to images/new_test")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-p', type=str, default='result/final_model.pt', help='Path to the trained model')
    parser.add_argument('--max_time_step', '-st', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--gpu', '-g', type=str, default='0', help='GPU index to use')
    args = parser.parse_args()
    
    main(args)