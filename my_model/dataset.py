import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path

## One source: https://averdones.github.io/reading-tabular-data-with-pytorch-and-training-a-multilayer-perceptron/

class TotalTurnoverDataset(Dataset):
    """Total Turnover dataset."""

    def __init__(self, pathdir):
        """Initializes instance of class TotalTurnoverDataset.
        Args:
            pathdir (str): Path to the npy file with the data.
        """
        self.X_num_train = torch.from_numpy(np.load(os.path.join(pathdir, 'X_num_train.npy'), allow_pickle=True))

    def __len__(self):
        return len(self.X_num_train)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.X_num_train[idx]
