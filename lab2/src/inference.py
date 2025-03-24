import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import torch
from utils import dice_score
from oxford_pet import load_dataset
import numpy as np
from torch.utils.data import DataLoader

def inference(args):
    dataset = load_dataset(args.data_path, mode = "test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    if args.model_type == 0:
        from models.unet import UNet
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == 1:
        from models.resnet34_unet import ResNet34_UNet
        model = ResNet34_UNet(in_channels=3, out_channels=1).to(device)
    else:
        raise ValueError("Invalid model choice")

    model.load_state_dict(torch.load(args.model, weights_only=True))

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    model.eval()

    return_input = []
    return_output = []
    return_label = []
    dice_scores = []
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            dice_scores.append(dice_score(outputs, labels).item())
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > 0.5).float()
            return_input.extend(inputs)
            return_output.extend(outputs)
            return_label.extend(labels)
    return return_input, return_output, return_label, np.array(dice_scores)




def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default=r'./saved_models/unet.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default=r"./dataset/oxford-iiit-pet", help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    parser.add_argument('--model_type', '-m', type=int, default=0, help='choose the model to inference: 0 for UNet, 1 for ResNet34_UNet')

    
    return parser.parse_args()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    args = get_args()
    input, output, label, dice_scores = inference(args)
    print(f"Average Dice Score: {np.mean(dice_scores)}")
    for _ in range(10):
        ind = random.randint(0, len(output)-1)
        imput_sample = input[ind].cpu().detach().numpy()
        output_sample = output[ind].cpu().detach().numpy()
        label_sample = label[ind].cpu().detach().numpy()

        input_sample = np.transpose(imput_sample, (1, 2, 0))
        output_sample = np.transpose(output_sample, (1, 2, 0))
        label_sample = np.transpose(label_sample, (1, 2, 0))

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(input_sample)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(output_sample)
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(label_sample)
        plt.title("Ground Truth Mask")
        plt.axis('off')
        plt.show()