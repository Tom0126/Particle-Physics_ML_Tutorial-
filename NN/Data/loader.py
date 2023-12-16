from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from  typing import Any
from Config.config import *
import numpy as np
import torch

class ImageSet(Dataset):
    def __init__(self,img_path,label_path,mean,std
                 ,mean_std_static,transform=None) -> None:
        super().__init__()
        datasets=np.load(img_path,allow_pickle=True)
        labels=np.load(label_path,allow_pickle=True)

        #standardize the train set
        if mean_std_static:
            mean=mean
            std=std
        else:
            mean = np.average(datasets)
            std = np.std(datasets)
        datasets = (datasets-mean)/std

        self.datasets = datasets.astype(np.float32)
        self.labels = labels.astype(np.longlong)
        self.transform = transform
    
    def __getitem__(self, index: Any):
        img = self.datasets[index]
        label = self.labels[index]
        if self.transform != None:
            img = self.transform(img)
        
        return img, label
    
    def __len__(self):
        return len(self.datasets)

def data_loader(img_path,
                label_path,
                mean:float=0.0,
                std:float=1.0,
                mean_std_static:bool=True,
                batch_size:int=32,
                shuffle:bool=False,
                num_workers:int=0):

    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset_train = ImageSet(img_path,
                             label_path,
                             mean=mean,
                             std=std,
                             mean_std_static=mean_std_static,
                             transform=transforms_train)
    loader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              drop_last=True)

    return loader_train





if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img_path = '/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Train/imgs.npy'
    label_path='/hpcfs/cepc/higgsgpu/siyuansong/PID/data/beam_2023/beam_xia/Train/labels.npy'
    loader = data_loader(img_path,label_path, num_workers=0,mean_std_static=True)
    for i, (img,label) in enumerate(loader):
        print('img:{} label:{}'.format(img.shape,label.shape))
        if i==0:
            break
