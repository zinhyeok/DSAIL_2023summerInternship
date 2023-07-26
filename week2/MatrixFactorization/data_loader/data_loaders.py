import os
import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import utils as utils

from torchvision import datasets, transforms
from base import BaseDataLoader

class MovieLensDataset(Dataset):
    #preporcessing
    """
    MovieLens dataset loading for CML
    """
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep='\t', header=None)
        # self.data.columns = ["user_id", "item_id", "rating", "timestamp"]
        #self.data = self.data.groupby(0).filter(lambda x: (x[2] == 1).sum() >= 10)        
        self.user = self.data[0]
        self.item = self.data[1]
        self.rating = self.data[2]
        self.timestamp = self.data[3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user = self.user[idx]
        item = self.item[idx]
        rating = self.rating[idx]
        timestamp = self.timestamp[idx]

        return user, item, rating, timestamp
                


class MovieLensDataLoader(BaseDataLoader):
    """
    Movielens data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = MovieLensDataset(os.path.join(data_dir, 'ml-100k/u.data'))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
