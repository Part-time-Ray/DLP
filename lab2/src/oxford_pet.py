import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import shutil
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from scipy.ndimage import convolve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        # return 32
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

def filter_mask(image):
    high_pass_kernel = np.array([[-0.0316, -0.0093, -0.0022, -0.0093, -0.0316],
                                  [-0.0093,  0.0078,  0.0127,  0.0078, -0.0093],
                                  [-0.0022,  0.0127,  0.0192,  0.0127, -0.0022],
                                  [-0.0093,  0.0078,  0.0127,  0.0078, -0.0093],
                                  [-0.0316, -0.0093, -0.0022, -0.0093, -0.0316]], dtype=np.float32)

    low_pass_kernel = np.array([[0.0751, 0.1238, 0.0751],
                                 [0.1238, 0.2042, 0.1238],
                                 [0.0751, 0.1238, 0.0751]], dtype=np.float32)
    
    re_image = np.zeros_like(image)

    for c in range(3):
        re_image[..., c] = convolve(image[..., c], low_pass_kernel, mode='nearest')
        re_image[..., c] = np.clip(re_image[..., c], 0, 255)
    
    return re_image

def remove_shadow(image):
    img_uint8 = image.astype(np.uint8)

    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    hsv_corrected = cv2.merge([h, s, v_eq])
    result = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2RGB)

    return result.astype(np.float32)

class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):
        sample = super().__getitem__(*args, **kwargs)

        image = Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR)
        mask = Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST)
        trimap = Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST)

        image = np.array(image, dtype=np.float32)
        # image = filetr_mask(image) # 會大幅增加訓練時間
        image = remove_shadow(image) # 會大幅增加訓練時間
        image = image / 255.0
        # image = filter_mask(remove_shadow(np.array(image, dtype=np.float32))) / 255.0
        mask = np.array(mask, dtype=np.float32)
        trimap = np.array(trimap, dtype=np.float32)

        mask = (mask > 0).astype(np.float32)

        if self.mode == "train" and np.random.rand() < 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
            trimap = np.fliplr(trimap).copy()
            


        image = torch.from_numpy(np.moveaxis(image, -1, 0))  # HWC -> CHW
        mask = torch.from_numpy(np.expand_dims(mask, 0))
        trimap = torch.from_numpy(np.expand_dims(trimap, 0))

        return image, mask, trimap


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)


def load_dataset(data_path, mode):
    if not os.path.exists(os.path.join(data_path, "images")) or not os.path.exists(os.path.join(data_path, "annotations")):
        print("Downloading Oxford-IIIT Pet Dataset...")
        OxfordPetDataset.download(data_path)

    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode)
    return dataset

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    data_path = os.path.join(".", "dataset", "oxford-iiit-pet")
    dataset = load_dataset(data_path, "test")
    
    for _ in range(10):
        img = np.transpose(dataset[random.randint(0, len(dataset)-1)][0], (1, 2, 0))
        plt.imshow(img)
        plt.axis('off') 
        plt.show()