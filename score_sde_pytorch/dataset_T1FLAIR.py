import os
import glob
import numpy as np
import torch

from torch.utils.data import Dataset


def get_loader(config):
    train_ds = Customdataset(config.data.data_root, 'train')
    valid_ds = Customdataset(config.data.data_root, 'test')

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=config.eval.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_dataloader, valid_dataloader


class Customdataset(Dataset):
    def __init__(self, data_root, mode):
        self.T2_data_path = glob.glob(os.path.join(data_root, mode, 'T1/*.npy'))
        self.T2_data_path = sorted(self.T2_data_path)

    def __len__(self):
        return len(self.T2_data_path)
        
    def __getitem__(self, index):
        T2_path = self.T2_data_path[index]
        T2_array = np.load(T2_path).reshape(1, 256, 256).astype(np.float32)
        T2_array = T2_array / 255.0
        T2_array = 2 * T2_array - 1

        FLAIR_path = T2_path.replace('/T1/', '/FLAIR/')
        FLAIR_array = np.load(FLAIR_path).reshape(1, 256, 256).astype(np.float32)
        FLAIR_array = FLAIR_array / 255.0
        FLAIR_array = 2 * FLAIR_array - 1

        input = np.concatenate([T2_array, FLAIR_array], 0)

        return input
