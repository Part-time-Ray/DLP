
import os
from glob import glob
import torch
from torch.utils.data import Dataset as torchData
import json
from torchvision.datasets.folder import default_loader as imgloader
from torchvision import transforms
from evaluator import evaluation_model
from dataloader import get_test_dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
])
test_file = os.path.join('images', 'test')
new_test_file = os.path.join('images', 'new_test')
label, new_label = get_test_dataset(device=device)
eval_model = evaluation_model()

image = torch.cat([transform(imgloader(os.path.join(test_file, f'{i}.png'))).unsqueeze(0) for i in range(len(os.listdir(test_file)))], dim=0)
new_image = torch.cat([transform(imgloader(os.path.join(new_test_file, f'{i}.png'))).unsqueeze(0) for i in range(len(os.listdir(new_test_file)))], dim=0)
image = image.to(device=device)
new_image = new_image.to(device=device)

score = eval_model.eval(image, label)
print("Test set accuracy:", score)

new_score = eval_model.eval(new_image, new_label)
print("New test set accuracy:", new_score)
