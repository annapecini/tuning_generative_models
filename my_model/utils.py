# import torch.utils.data
from torch.utils.data import DataLoader, random_split
from dataset import TotalTurnoverDataset
import os
import numpy as np
import pandas as pd
from pathlib import Path


# def gather(consts: torch.Tensor, t: torch.Tensor):
#     """Gather consts for $t$ and reshape to feature map shape"""
#     c = consts.gather(-1, t)
#     return c.reshape(-1, 1, 1, 1)

def get_data(args):
    # Load dataset
    dataset = TotalTurnoverDataset(args.dataset_path)

    # Split into training and test
    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # trainset, valset = random_split(dataset, [train_size, val_size])

    # Dataloaders
    trainloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # valloader = DataLoader(valset, batch_size=200, shuffle=False)

    return trainloader#, valloader


def save_images(images, path, epoch,**kwargs):
    path.mkdir(exist_ok=True)
    np.save(images, os.path.join(path, f"{epoch}.npy"))


def save_synthetic_data(x, path="synthetic_my_model.csv"):
    # Save sampled data
    df = pd.DataFrame(x, columns=['date',
                                  'whi_temp',
                                  'whi_temp_min',
                                  'whi_temp_max',
                                  'whi_feels_like',
                                  'whi_pressure',
                                  'whi_humidity',
                                  'whi_clouds',
                                  'whi_rain'])

    # Convert number of days to date
    df["date"] = pd.to_datetime(df["date"], unit="d").dt.date

    # Sort by date
    df.sort_values(by=['date'], inplace=True)

    # Save dataframe as csv
    # df.to_csv(path, index=True)
    df.to_csv(path, index=False)
