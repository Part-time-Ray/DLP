import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10



def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        self.iteration = current_epoch
        self.args = args
        if args.kl_anneal_type == "Cyclical":
            self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=args.kl_anneal_cycle, ratio=args.kl_anneal_ratio)
        elif args.kl_anneal_type == "Monotonic":
            self.frange_cycle_linear(args.num_epoch, start=0.0, stop=1.0, n_cycle=args.num_epoch, ratio=args.kl_anneal_ratio)
        
    def update(self):
        self.iteration += 1
    
    def get_beta(self):
        if self.args.kl_anneal_type == "None": return 1.0
        return round(self.beta[self.iteration], 2)

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        self.beta = []
        cycle_num = n_iter // n_cycle
        for i in range(cycle_num+1):
            for j in range(n_cycle):
                val = start + (stop - start) * ratio * (j+1)
                self.beta.append(min(stop, val))

def check_loss_line(train_loss, val_loss, scores, save_dir=None, title_prefix='Loss'):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} - Training')
    plt.grid(True)
    try: plt.ylim(0, min(np.max(train_loss), 0.08))
    except: pass
    if save_dir:
        plt.savefig(f'{save_dir}/train_loss.png')
    plt.show()
    plt.close()


    plt.figure(figsize=(8, 6))
    plt.plot(epochs, val_loss, label='Validation Loss', color='green', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title_prefix} - Validation')
    plt.grid(True)
    if save_dir:
        plt.savefig(f'{save_dir}/val_loss.png')
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, scores, label='PSNR', color='orange', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR score')
    plt.title('PSNR score Validation')
    plt.grid(True)
    plt.text(0.95, 0.95, f'Max: {max(scores):.2f}', ha='right', va='top',
             transform=plt.gca().transAxes, fontsize=10, color='red')
    if save_dir:
        plt.savefig(f'{save_dir}/psnr_score.png')
    plt.show()
    plt.close()
    

class SmartSave:
    def __init__(self, best_loss, best_score, loss_path, socre_path):
        self.best_loss = best_loss
        self.best_score = best_score
        self.loss_path = loss_path
        self.score_path = socre_path
        print(f"Initial Best Loss: {best_loss:.4f}")
        print(f"Initial Best Score: {best_score:.4f}")

    def __call__(self, loss, score, model):
        if loss < self.best_loss:
            self.best_loss = loss
            self.save(model, self.loss_path)
        if score > self.best_score:
            self.best_score = score
            self.save(model, self.score_path)
    
    def save(self, model, path):
        torch.save({
            "state_dict": model.state_dict()
        }, path)
        print(f"=====save ckpt to {path}=====")

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 1
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.smart_save = SmartSave(np.inf, -np.inf, os.path.join(args.save_root, "best_loss.ckpt"), os.path.join(args.save_root, "best_score.ckpt"))
        
    def forward(self, img, pre_img, label, batch_size):
        img_vector = self.frame_transformation(img)
        pre_img_vector = self.frame_transformation(pre_img)
        label_vector = self.label_transformation(label)
        z, mu, logvar = self.Gaussian_Predictor(pre_img_vector, label_vector)
        re_gen_img = self.Generator(self.Decoder_Fusion(pre_img_vector, label_vector, z))
        mse_loss = self.mse_criterion(re_gen_img, img) 
        kl_loss = kl_criterion(mu, logvar, batch_size)
        return mse_loss, kl_loss, re_gen_img

    
    def training_stage(self):
        training_loss, validation_loss, scores = [], [], []


        for epoch in range(self.args.num_epoch_warmup):
            train_loader = self.train_dataloader()
            train_loss = 0
            for i, (img, label) in enumerate(pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.pretrain_one_step_decoder(img.clone(), label.clone())
                train_loss += loss.detach().cpu()
                pbar.set_description(f"Epoch {epoch+1}/{self.args.num_epoch_warmup}, WarmUp Loss: {train_loss/(i+1)}" , refresh=False)
            train_loss = train_loss / len(train_loader)

        # for param in self.frame_transformation.parameters():
        #     param.requires_grad = False

        # for param in self.label_transformation.parameters():
        #     param.requires_grad = False
        
        # for param in self.Decoder_Fusion.parameters():
        #     param.requires_grad = False

        # for param in self.Generator.parameters():
        #     param.requires_grad = False

        
        print(f"Training Epochs: {self.args.num_epoch}")
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            train_loss = 0
            for i, (img, label) in enumerate(pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img.clone(), label.clone(), adapt_TeacherForcing)
                train_loss += loss.detach().cpu()

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    # pbar.set_description(f"\t(TF:ON) Epoch {self.current_epoch}/{self.args.num_epoch}, Training Loss: {train_loss/(i)}" , refresh=False)
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, train_loss/(i), lr=self.scheduler.get_last_lr()[0])
                else:
                    # pbar.set_description(f"\t(TF:OFF) Epoch {self.current_epoch}/{self.args.num_epoch}, Training Loss: {train_loss/(i)}" , refresh=False)
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, train_loss/(i), lr=self.scheduler.get_last_lr()[0])
            train_loss = train_loss / len(train_loader)
            eval_loss, score = self.evaluate()
            scores.append(score)
            # if self.current_epoch % self.args.per_save == 0:
            #     self.save(os.path.join(self.args.save_root, f"check_point.ckpt"))
                # self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
            print(f"\nEpoch {self.current_epoch}/{self.args.num_epoch}, Average Training Loss: {train_loss:.4f}")
            print(f"Epoch {self.current_epoch}/{self.args.num_epoch}, Average Validation Loss: {eval_loss:.4f}")
            print(f"Average PSNR: {score:.4f}")
            self.smart_save(eval_loss, score, self)
            print()

            training_loss.append(train_loss)
            validation_loss.append(eval_loss)
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()
            check_loss_line(training_loss, validation_loss, scores, self.args.save_root)
            
            
    @torch.no_grad()
    def evaluate(self):
        val_loader = self.val_dataloader()
        total_loss = 0
        scores = []
        for i, (img, label) in enumerate(pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, score = self.val_one_step(img.clone(), label.clone())
            total_loss += loss.detach().cpu()
            scores.append(score)
            pbar.set_description(f"Epoch {self.current_epoch}/{self.args.num_epoch}, Validation Loss: {total_loss/(i+1)}" , refresh=False)
            # self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        avg_loss = total_loss / len(val_loader)
        return avg_loss, np.mean(scores)
    
    def pretrain_one_step_decoder(self, img, label):
        self.train()
        mse_loss = 0
        frames = img.shape[1]
        for f in range(1, frames):
            now_img = img[:, f]
            pre_img = img[:, f-1]
            pre_img_vector = self.frame_transformation(pre_img)
            label_vector = self.label_transformation(label[:, f])
            z = torch.randn(pre_img_vector.shape[0], self.args.N_dim, pre_img_vector.shape[2], pre_img_vector.shape[3]).to(self.args.device)
            re_gen_img = self.Generator(self.Decoder_Fusion(pre_img_vector, label_vector, z))
            mse_loss += self.mse_criterion(re_gen_img, now_img)
        self.optim.zero_grad()
        mse_loss.backward()
        self.optimizer_step()
        return mse_loss.detach()
    
    def training_one_step(self, img, label, adapt_TeacherForcing):
        self.train()
        beta = self.kl_annealing.get_beta()
        mse_loss, kl_loss = 0, 0
        batch_size = img.shape[0]
        frames = img.shape[1]
        x_hat = img[:, 0]
        for f in range(1, frames):
            now_img = img[:, f]
            pre_img = img[:, f-1] if adapt_TeacherForcing else x_hat.detach()
            loss_temp, kl_loss_temp, re_gen_img = self(now_img, pre_img, label[:, f], batch_size)
            mse_loss += loss_temp
            kl_loss += kl_loss_temp
            x_hat = re_gen_img
        loss = mse_loss + beta * kl_loss
        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()
        return loss.detach()

        # beta = self.kl_annealing.get_beta()
        # loss = 0
        # for i in range(img.shape[0]):
        #     temp_img = img[i]
        #     x_hat = temp_img[0].unsqueeze(0)
        #     mse_loss, kl_loss = 0, 0
        #     for t in range(1, img.shape[1]):
        #         now_img = temp_img[t].unsqueeze(0)
        #         pre_img = temp_img[t-1].unsqueeze(0) if adapt_TeacherForcing else x_hat.detach()
        #         loss_temp, kl_loss_temp, re_gen_img = self(now_img, pre_img, label[i][t].unsqueeze(0), img.shape[0])
        #         mse_loss += loss_temp
        #         kl_loss += kl_loss_temp
        #         x_hat = re_gen_img
        #     loss += mse_loss / img.shape[0] + beta * kl_loss
        # self.optim.zero_grad()
        # loss.backward()
        # self.optimizer_step()
        # return loss.detach()
    
    def val_one_step(self, img, label):
        self.eval()
        assert img.shape[0] == 1, "Batch size should be 1 in validation stage"
        batch_size = img.shape[0]
        frames = img.shape[1]
        beta = self.kl_annealing.get_beta()
        mse_loss, kl_loss = 0, 0
        x_hat = img[:, 0]
        gif = [img[0, 0].permute(1,2,0).detach().cpu().numpy()]
        score = []
        for f in range(1, frames):
            now_img = img[:, f]
            pre_img = x_hat.detach()
            loss_temp, kl_loss_temp, re_gen_img = self(now_img, pre_img, label[:, f], batch_size)
            mse_loss += loss_temp
            kl_loss += kl_loss_temp
            x_hat = re_gen_img
            gif.append(re_gen_img[0].permute(1,2,0).detach().cpu().numpy())
            score.append(Generate_PSNR(x_hat.detach(), img[:, f]).item())
        self.make_gif(gif, "val.gif")
        # loss = mse_loss + beta * kl_loss
        loss = mse_loss
        return loss.detach(), np.mean(score)


        # beta = self.kl_annealing.get_beta()
        # loss = 0
        # for i in range(img.shape[0]):
        #     temp_img = img[i]
        #     x_hat = temp_img[0].unsqueeze(0)
        #     mse_loss, kl_loss = 0, 0
        #     for t in range(1, img.shape[1]):
        #         now_img = temp_img[t].unsqueeze(0)
        #         pre_img = x_hat.detach()
        #         loss_temp, kl_loss_temp, re_gen_img = self(now_img, pre_img, label[i][t].unsqueeze(0), img.shape[0])
        #         mse_loss += loss_temp
        #         kl_loss += kl_loss_temp
        #         x_hat = re_gen_img
        #     loss += mse_loss / img.shape[0] + beta * kl_loss
        # return loss.detach()
                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(os.path.join(self.args.save_root, img_name), format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch < self.tfr_sde:
            return
        self.tfr = max(0, self.tfr - self.tfr_d_step)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    # def save(self, path):
    #     torch.save({
    #         "state_dict": self.state_dict(),
    #         "optimizer": self.optim.state_dict(),  
    #         "lr"        : self.scheduler.get_last_lr()[0],
    #         "tfr"       :   self.tfr,
    #         "last_epoch": self.current_epoch
    #     }, path)
    #     print(f"save ckpt to {path}")

    def load_checkpoint(self):
        pass

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 0.1)
        self.optim.step()



def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    
    device = args.device
    if args.device == "cuda":
        args.device = args.device + ":" + str(args.gpu)

    model = VAE_Model(args)
    model.to(args.device)
    model.load_checkpoint()
    if args.test:
        model.evaluate()
    else:
        model.training_stage()




# python ./Lab4_template/Trainer.py --ckpt_path ./save_path/check_point.ckpt
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=5)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=0)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, default='./LAB4_Dataset/LAB4_Dataset', help="Your Dataset Path") # required
    parser.add_argument('--save_root',     type=str, default='./save_path', help="The path to save your data") # required
    parser.add_argument('--num_workers',   type=int, default=12)
    parser.add_argument('--num_epoch',     type=int, default=300,     help="number of total epoch")
    parser.add_argument('--num_epoch_warmup',    type=int, default=5,       help="number of total epoch warmup")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=0.5,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=8,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str, default="./save_path", help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical', choices=['Cyclical', 'Monotonic', 'None'],       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=30,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=0.05,              help="") 

    args = parser.parse_args()
    
    main(args)
