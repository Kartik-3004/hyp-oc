import csv
import pandas as pd
import numpy as np
from skimage import io
import PIL
import torch
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import random
import os
import glob

class ROSEYoutu(Dataset):
    def __init__(self, split, csv_root, data_root):
        super().__init__()
        self.data_root = data_root
        self.csv_root = csv_root
        self.transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            CenterCrop((224,224)),
            ToTensor(),
            Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]),
        ])
        if split == "train":
            self.data_info = pd.read_csv(os.path.join(csv_root, 'ROSEYoutu/train.csv'), header=None)
        else:
            self.data_info = pd.read_csv(os.path.join(csv_root, 'ROSEYoutu/test.csv'), header=None)
        self.img_path = np.asarray(self.data_info.iloc[1:,0])
        self.label_arr = np.asarray(self.data_info.iloc[1:,1])
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        img_name = self.img_path[index]
        image_path = os.path.join(self.data_root, img_name)
        img_as_img = Image.open(image_path)
        img_as_img = img_as_img.convert("RGB")
        img_cropped = self.transform(img_as_img)
        label = int(self.label_arr[index])
        return img_cropped, label
    
    def __len__(self):
        return self.data_len

class ReplayAttack(Dataset):
    def __init__(self, split, csv_root, data_root):
        super().__init__()
        self.data_root = data_root
        self.csv_root = csv_root
        self.transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            CenterCrop((224,224)),
            ToTensor(),
            Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]),
        ])
        if split == "train":
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"ReplayAttack/train.csv"), header=None)
        elif split == "test":
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"ReplayAttack/test.csv"), header=None)
        elif split == "val":
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"ReplayAttack/eval.csv"), header=None)
        self.img_path = np.asarray(self.data_info.iloc[1:,0])
        self.label_arr = np.asarray(self.data_info.iloc[1:,1])
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        img_name = self.img_path[index]
        image_path = os.path.join(self.data_root, img_name)
        img_as_img = Image.open(image_path)
        img_as_img = img_as_img.convert("RGB")
        img_cropped = self.transform(img_as_img)
        label = int(self.label_arr[index])
        return img_cropped, label
    
    def __len__(self):
        return self.data_len

class CASIA_MFSD(Dataset):
    def __init__(self, split, csv_root, data_root):
        super().__init__()
        self.data_root = data_root
        self.csv_root = csv_root
        self.transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            CenterCrop((224,224)),
            ToTensor(),
            Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]),
        ])
        if split == "train":
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"CASIA_MFSD/train.csv"), header=None)
        else:
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"CASIA_MFSD/test.csv"), header=None)
        self.img_path = np.asarray(self.data_info.iloc[1:,0])
        self.label_arr = np.asarray(self.data_info.iloc[1:,1])
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        img_name = self.img_path[index]
        image_path = os.path.join(self.data_root, img_name)
        img_as_img = Image.open(image_path)
        img_as_img = img_as_img.convert("RGB")
        img_cropped = self.transform(img_as_img)
        label = int(self.label_arr[index])
        return img_cropped, label
    
    def __len__(self):
        return self.data_len

class MSU_MFSD(Dataset):
    def __init__(self, split, csv_root, data_root):
        super().__init__()
        self.data_root = data_root
        self.csv_root = csv_root
        self.transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            CenterCrop((224,224)),
            ToTensor(),
            Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]),
        ])
        if split == "train":
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"MSU_MFSD/train.csv"), header=None)
        else:
            self.data_info = pd.read_csv(os.path.join(self.csv_root,"MSU_MFSD/test.csv"), header=None)
        self.img_path = np.asarray(self.data_info.iloc[1:,0])
        self.label_arr = np.asarray(self.data_info.iloc[1:,1])
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        img_name = self.img_path[index]
        image_path = os.path.join(self.data_root, img_name)
        img_as_img = Image.open(image_path)
        img_as_img = img_as_img.convert("RGB")
        img_cropped = self.transform(img_as_img)
        label = int(self.label_arr[index])
        return img_cropped, label
    
    def __len__(self):
        return self.data_len

class OULU_NPU(Dataset):
    def __init__(self, split, csv_root, data_root):
        super().__init__()
        self.protocol_num = 1
        self.data_root = data_root
        self.csv_root = csv_root
        self.transform = Compose([
            Resize((224,224), interpolation=BICUBIC),
            CenterCrop((224,224)),
            ToTensor(),
            Normalize(mean=[129.186279296875, 104.76238250732422, 93.59396362304688], std=[1, 1, 1]),
        ])
        if split == "train":
            self.data_dir = os.path.join(self.data_root, "OULU_NPU/Train_frames")
            self.protocol_dir = os.path.join(self.csv_root, f"OULU_NPU/Protocols/Protocol_{self.protocol_num}/Train.txt")
        elif split == "test":
            self.data_dir = os.path.join(self.data_root,"OULU_NPU/Test_frames")
            self.protocol_dir = os.path.join(self.csv_root,f"OULU_NPU/Protocols/Protocol_{self.protocol_num}/Test.txt")
        elif split == "val":
            self.data_dir = os.path.join(self.data_root,"OULU_NPU/Dev_frames")
            self.protocol_dir = os.path.join(self.csv_root,f"OULU_NPU/Protocols/Protocol_{self.protocol_num}/Dev.txt")
        self.data = []
        self.labels = []
        
        file_n = os.path.join(self.protocol_dir)
        with open(file_n, 'r') as file:
            lines = file.readlines()
            for line in lines:
                label, filename = line.rstrip().split(",")
                folder_name = os.path.join(self.data_dir, filename)
                filenames = os.listdir(folder_name)
                image_paths = []
                for filename in filenames:
                    image_paths.append(os.path.join(folder_name, filename))
                labels_arr = []
                if split == "train":
                    if label == "+1":
                        labels_arr.extend([0]*len(image_paths))
                        self.data.extend(image_paths)
                        self.labels.extend(labels_arr)
                else:
                    if label == "+1":
                        labels_arr.extend([0]*len(image_paths))
                        self.data.extend(image_paths)
                        self.labels.extend(labels_arr)
                    else:
                        labels_arr.extend([1]*len(image_paths))                    
                        self.data.extend(image_paths)
                        self.labels.extend(labels_arr)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data[idx]
        label = self.labels[idx]
        img_as_img = Image.open(image_path)
        img_as_img = img_as_img.convert("RGB")
        image = self.transform(img_as_img)        
        return image, label
