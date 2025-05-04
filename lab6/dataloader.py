
import os
from glob import glob
import torch
from torch.utils.data import Dataset as torchData
import json
from torchvision.datasets.folder import default_loader as imgloader
from torchvision import transforms

class Dataset(torchData):
    def __init__(self, mode='train', folder_path='iclevr'):
        super().__init__()
        self.mode = mode
        assert mode in ['train', 'val', 'test', 'new_test'], "There is no such mode !!!"
        if mode in ['test', 'new_test']:
            with open('test.json', 'r', encoding='utf-8') as file:
                datas = json.load(file)
            with open('new_test.json', 'r', encoding='utf-8') as file:
                new_datas = json.load(file)
            with open('objects.json', 'r', encoding='utf-8') as file:
                objects = json.load(file)
            label = [[1 if obj in data else 0 for obj in objects] for data in datas]
            new_label = [[1 if obj in data else 0 for obj in objects] for data in new_datas]
            self.label = label if mode == 'test' else new_label
            return

        self.folder_path = folder_path
        with open('train.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        with open('objects.json', 'r', encoding='utf-8') as file:
            self.objects = json.load(file)
        keys = list(data.keys())
        values = list(data.values())
        total_len = len(keys)
        train_len = int(total_len/3 * 0.9)*3

        if mode == 'train':
            self.data = keys[:train_len]
            self.label = values[:train_len]
        elif mode == 'val':
            self.data = keys[train_len:]
            self.label = values[train_len:]

        self.data = [os.path.join(self.folder_path, fname) for fname in self.data]
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
        ])
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        if self.mode in ['test', 'new_test']:
            return torch.tensor(self.label[index])
        
        data = self.transform(imgloader(self.data[index]))
        label = torch.tensor([1 if obj in self.label[index] else 0 for obj in self.objects])
        return data, label
    


def get_test_dataset(device='cuda'):
    with open('test.json', 'r', encoding='utf-8') as file:
        datas = json.load(file)
    with open('new_test.json', 'r', encoding='utf-8') as file:
        new_datas = json.load(file)
    with open('objects.json', 'r', encoding='utf-8') as file:
        objects = json.load(file)
    label = [[1 if obj in data else 0 for obj in objects] for data in datas]
    new_label = [[1 if obj in data else 0 for obj in objects] for data in new_datas]
    return torch.tensor(label, device=device), torch.tensor(new_label, device=device)
    

if __name__ == '__main__':
    import random
    from torchvision.utils import save_image
    from utils import BetaScheduler

    dataset = Dataset(mode='train', folder_path='iclevr')
    ind = random.randint(0, len(dataset)-1)
    data, label = dataset[ind]
    print(label.tolist())
    with open('objects.json', 'r', encoding='utf-8') as file:
        objects = json.load(file)
    for name in objects:
        if label[objects[name]]:
            print(name)
    
    datas = []
    beta_scheduler = BetaScheduler(num_diffusion_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda')
    for i in range(0, 20):
        data, noise = beta_scheduler.make_noise(data, i)
        datas.append(data)
    save_image(torch.stack(datas), os.path.join('temp.png'), normalize=True)


