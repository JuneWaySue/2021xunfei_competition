from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import pandas as pd


class ImageDataSet(Dataset):
    '''图片加载和处理'''
    
    def __init__(self,df,transform):
        self.df=df
        self.transform=transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx].path
        image_id = self.df.iloc[idx].image_id
        order = self.df.iloc[idx].category_id
        if self.df.iloc[idx].is_train == 1:
            image = image_path
        else:
            image = Image.open(image_path).convert('RGBA').convert('RGB')
        vector=np.zeros(137,dtype=float)
        vector[order]=1.0
        label = torch.from_numpy(vector)
        image=self.transform(image)
        return image, label, order, image_id
    
class ImageDataSet2(Dataset):
    '''图片加载和处理'''
    
    def __init__(self,df,transform):
        self.df=df
        self.transform=transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = self.df.iloc[idx].path
        image_id = self.df.iloc[idx].image_id
        order = self.df.iloc[idx].category_id
        image = Image.open(image_path).convert('RGBA').convert('RGB')
        label = torch.from_numpy(np.array(order))
        image=self.transform(image)
        return image, label, order, image_id