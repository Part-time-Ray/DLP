import torch
from torch.utils.data import DataLoader
import oxford_pet
import utils

def evaluate(net, data_loader, device):
    net.eval()
    total_dice_loss = 0
    num_samples = 0
    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            dice_loss = utils.dice_loss(outputs, labels)
            total_dice_loss += dice_loss.sum().item()
            num_samples += 1
    avg_loss = total_dice_loss / num_samples
    print(f"\tEvaluate Dice Loss: {avg_loss}")
    return avg_loss