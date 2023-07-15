import os
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from torchvision import datasets, transforms
from base import BaseDataLoader

class MovieLensDataset(Dataset):
    """
    MovieLens dataset loading class
    """
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep='\t', header=None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_info = self.data.iloc[idx].values
        user_id = int(user_info[0])
        item_id = int(user_info[1])
        rating = self.__preprocess_rating(user_info[2])
        timestamp = user_info[3].astype('float32')
        return user_id, item_id, rating, timestamp
    
    #change rate to implicit data
    def __preprocess_rating(self, rating):
        rating[rating <= 3] = 0
        rating[rating > 3] = 1

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MovieLensDataLoader(BaseDataLoader):
    """
    Movielens data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MovieLensDataset(os.path.join(data_dir, 'u.user'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)