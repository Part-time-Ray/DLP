import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# def dice_score(output, target, epsilon=0.1):
#     assert output.shape[1] == 1
#     assert target.shape[1] == 1
#     pred = torch.sigmoid(output)
#     intersection = (pred * target).sum(dim=(1, 2, 3))
#     union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
#     dice = (2. * intersection + epsilon) / (union + epsilon)
#     return dice.mean()

def dice_score(pred_mask, gt_mask, epsilon=0.1):
    # implement the Dice score here
    # print(gt_mask.shape)
    assert pred_mask.shape[1] == 1
    assert pred_mask.shape[2] == 256
    assert pred_mask.shape[3] == 256
    assert pred_mask.shape == gt_mask.shape, "Predicted mask and ground truth mask should have the same shape"
    pred_mask = torch.sigmoid(pred_mask)
    # pre_mask is already convert to 0 or 1 before calling this function
    intersection = ((pred_mask * gt_mask) + ((1-pred_mask)*(1-gt_mask))).sum(dim=(1, 2, 3))
    # print(intersection)

    width = pred_mask.shape[2]
    height = pred_mask.shape[3]
    score = (2 * intersection + epsilon) / (width * height * 2 + epsilon)
    return score.mean()

def dice_loss(output, target, epsilon=0.1):
    return 1 - dice_score(output, target, epsilon)


def check_loss_line(train_loss, val_loss, save_path=None, title='Training and Validation Loss'):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Training Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # plt.minorticks_on()
    # plt.grid(True, linestyle='--', alpha=0.7, which='both')

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
