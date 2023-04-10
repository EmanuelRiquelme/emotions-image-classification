import torch
from torch.utils.data import Dataset
from  torchvision import transforms
import os
from PIL import Image

class Emotions(Dataset):
    def __init__(self,root_dir = 'test',transform = None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Resize((192,192)),
            transforms.RandomHorizontalFlip(p=.3),
            transforms.RandomRotation((0,180)),
            transforms.Normalize(mean = .5,std = .25),
        ])


    def __getlabels__(self):
        labels = [file for file in os.listdir(self.root_dir)]
        return dict(zip(labels,torch.arange(len(labels))))

    def __getfiles__(self):
        sub_dirs = [file for file in os.listdir(self.root_dir)]
        img_names = []
        for sub_dir in sub_dirs:
            img_names.extend(f'{self.root_dir}/{sub_dir}/{file}' for file in os.listdir(f'{self.root_dir}/{sub_dir}'))
        return img_names

    def __len__(self):
        return len(self.__getfiles__())
    
    def __getitem__(self,idx):
        file_name = self.__getfiles__()[idx]
        label = file_name.split('/')[1]
        label = self.__getlabels__()[label]
        img = self.transform(Image.open(file_name))
        return img,label
