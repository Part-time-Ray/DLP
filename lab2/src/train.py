import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from evaluate import evaluate
from oxford_pet import load_dataset
import utils

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")

    # load data
    dataset = load_dataset(args.data_path, mode = "train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataset = load_dataset(args.data_path, mode = "valid")
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    if args.model == 0:
        net = 'unet'
        path = os.path.join(args.save_path, f'{net}.pth')
        model = UNet(in_channels=3, out_channels=1).to(device)

        # hyperparameters
        criterion = utils.dice_loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.7)
        early_stopping = utils.EarlyStopping(patience=10, min_delta=0.001, path=path)
    elif args.model == 1:
        net = 'ResNet34_unet'
        path = os.path.join(args.save_path, f'{net}.pth')
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)

        # hyperparameters
        criterion = utils.dice_loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.7)
        early_stopping = utils.EarlyStopping(patience=10, min_delta=0.001, path=path)
    else:
        raise ValueError("Invalid model choice")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    
    # Training loop
    train_loss = []
    validation_loss = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for images, masks, _ in dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss / len(dataloader)}")
        train_loss.append(epoch_loss / len(dataloader))
        validation_loss.append(evaluate(model, val_dataloader, device))
        scheduler.step()
        early_stopping(validation_loss[-1], model)
        if early_stopping.early_stop:
            print("Early stopping...")
            break

    utils.check_loss_line(train_loss, validation_loss, save_path=rf'./src/result/{net}.png', title=f'training and validation loss with {net}')
    # Alraeady save the best model in the early stopping
    # torch.save(model.state_dict(), path)

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default=r"./dataset/oxford-iiit-pet", help='path of the input data')
    parser.add_argument('--save_path', '-s', type=str, default=r"./saved_models", help='save path of the model')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model', '-m', type=int, default=0, help='choose the model to train: 0 for UNet, 1 for ResNet34_UNet')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)