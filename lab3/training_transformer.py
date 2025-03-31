import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
from utils import check_loss_line
import yaml
from torch.utils.data import DataLoader

#TODO2 step1-4: design the transformer training strategy
class SmartSave:
    def __init__(self, best_loss):
        self.best_loss = best_loss
        print(f"Initial Best Loss: {best_loss:.4f}")
    def __call__(self, path, model, optim, scheduler, epoch, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.save(path, model, optim, scheduler, epoch, loss)
    
    def save(self, path, model, optim, scheduler, epoch, loss):
        transformer_state_dict = (
            model.module.transformer.state_dict() 
            if isinstance(model, nn.DataParallel) 
            else model.transformer.state_dict()
        )
        torch.save({
            'transformer_state_dict': transformer_state_dict,
            'optimizer_state_dict': optim.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'loss': loss,
        }, path)
        print(f"Checkpoint saved at {path}")
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.device = args.device
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"])
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device=self.device)
        self.optim, self.scheduler = self.configure_optimizers()
        self.start_epoch = 1
        best_loss = np.inf
        if os.path.exists(args.checkpoint_path):
            checkpoint = torch.load(args.checkpoint_path, weights_only=False)
            if isinstance(self.model, nn.DataParallel):
                self.model.module.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            else:
                self.model.transformer.load_state_dict(checkpoint['transformer_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss']
        self.prepare_training()
        self.smart_save = SmartSave(best_loss) 
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)
    def train_one_epoch(self, train_loader, epoch, args):
        self.model.train()
        train_loss = []
        tq = tqdm(train_loader)
        for i, images in enumerate(tq):
            images = images.to(args.device)
            logits, z_indices = self.model(images)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
            loss.backward()
            train_loss.append(loss.item())
            if (i + 1) % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            tq.set_description(f"Epoch {epoch}/{args.epochs}, Training Loss: {np.mean(train_loss):.6f}")
        return np.mean(train_loss)

    def eval_one_epoch(self, val_loader, epoch, args):
        self.model.eval()
        eval_loss = []
        with torch.no_grad():
            tq = tqdm(val_loader)
            for images in tqdm(tq):
                images = images.to(args.device)
                logits, z_indices = self.model(images)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), z_indices.reshape(-1))
                eval_loss.append(loss.item())
                tq.set_description(f"Epoch {epoch}/{args.epochs}, Validation Loss: {np.mean(eval_loss):.6f}")
        avg_loss = np.mean(eval_loss)
        self.smart_save(args.checkpoint_path, self.model, self.optim, self.scheduler, epoch, avg_loss)
        # print(f"Epoch {epoch}/{args.epochs}, Validation Loss: {avg_loss:.6f}")
        return avg_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.7,patience=10, min_lr=5e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        scheduler = None
        return optimizer,scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./transformer_checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=1, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    # parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
#TODO2 step1-5:    
    training_loss = []
    validation_loss = []
    for epoch in range(train_transformer.start_epoch, args.epochs + 1):
        # print(f"Epoch {epoch}/{args.epochs}")

        # Training phase
        avg_train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        training_loss.append(avg_train_loss)

        # Validation phase
        avg_val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        validation_loss.append(avg_val_loss)

        check_loss_line(training_loss, validation_loss, "./loss_line.jpg", title='Training and Validation Loss')

        # Scheduler step (if using scheduler)
        if train_transformer.scheduler:
            train_transformer.scheduler.step(avg_val_loss)