import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio
# from unet import UNet
# from unet_2 import UNet
from unet_3 import UNet
from dataloader import Dataset
from utils import BetaScheduler, SMARTSave, inference
import matplotlib.pyplot as plt
from math import log10
from diffusers.optimization import get_cosine_schedule_with_warmup


# implement diffusion model
def main(args):
    train_dataset = Dataset(mode='train', folder_path='iclevr')
    eval_dataset = Dataset(mode='val', folder_path='iclevr')
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=True)
    val_loader = DataLoader(eval_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    beta_scheduler = BetaScheduler(num_diffusion_timesteps=args.max_time_step, beta_start=1e-4, beta_end=0.02, device=device)
    # model = UNet(args.max_time_step, in_channels=3, out_channels=3).to(device) # 24 channels for 24 classes
    model = UNet()
    if args.load_path != '':
        print(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path))
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DataParallel.")
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler =  get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps= len(train_loader) * 500,
    )
    criterion = nn.MSELoss()
    smart_save = SMARTSave(postfix = args.postfix)

    for epoch in range(args.num_epoch):
        model.train()
        tq = tqdm(train_loader, ncols=args.ncols)
        total_loss = 0
        for i, (x, label) in enumerate(tq):
            x = x.to(device)
            label = label.to(device)
            t = torch.randint(0, args.max_time_step, (x.size(0),), device=device).long()
            x_t, noise = beta_scheduler.make_noise(x, t)
            pred_noise = model(x_t, label, t)
            loss = criterion(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            tq.set_description(f"Epoch [{epoch+1}/{args.num_epoch}], loss: {total_loss / (i + 1):.6f}")

        if (epoch + 1) % 2 == 0:
            model.eval()
            total_loss = 0
            with torch.no_grad():
                tq = tqdm(val_loader, ncols=args.ncols)
                for i, (x, label) in enumerate(tq):
                    x = x.to(device)
                    label = label.to(device)
                    t = torch.randint(0, args.max_time_step, (x.size(0),), device=device).long()
                    x_t, noise = beta_scheduler.make_noise(x, t)
                    pred_noise = model(x_t, label, t)
                    loss = criterion(pred_noise, noise)
                    total_loss += loss.item()
                    tq.set_description(f"[Eval] loss: {total_loss / (i + 1)}")
            avg_loss = total_loss / len(val_loader)
            smart_save(model.module if isinstance(model, nn.DataParallel) else model, avg_loss)
        if (epoch + 1) % 10 == 0:
            gt, label = eval_dataset[random.randint(0, len(eval_dataset)-1)]
            label = label.unsqueeze(0).to(device)
            img = inference(model, beta_scheduler, label, image_size=(3, 64, 64), device=device)
            checkpoint_path = os.path.join('result', ('checkpoint' if args.postfix == '' else 'checkpoint' + '_' + args.postfix))
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
            save_image(img, os.path.join(f'inference{('_' if args.postfix != '' else '') + args.postfix}.png'), normalize=True)
            save_image(gt, os.path.join(f'ground_truth{('_' if args.postfix != '' else '') + args.postfix}.png'), normalize=True)
    # save the model
    torch.save(model.state_dict(), os.path.join('result', f'final_model{('_' if args.postfix != '' else '') + args.postfix}.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', '-b',     type=int,    default=32)
    parser.add_argument('--load_path', '-lp',     type=str,    default='')
    parser.add_argument('--lr', '-lr',            type=float,  default=1e-4,     help="initial learning rate")
    parser.add_argument('--device',               type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--gpu', '-g',            type=str, default='0')
    parser.add_argument('--num_workers',          type=int, default=12)
    parser.add_argument('--num_epoch',            type=int, default=500,     help="number of total epoch")
    parser.add_argument('--max_time_step', '-st', type=int, default=1000,     help="number of diffusion steps")    
    parser.add_argument('--postfix', '-pf',       type=str, default='', help="postfix for the model name")
    parser.add_argument('--ncols', '-n',          type=int, default=100, help="number of columns for tqdm")
    args = parser.parse_args()
    
    main(args)