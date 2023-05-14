import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import MLPDiffusion, LSTMDiffusion
import logging
import argparse
import pandas as pd
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, num_features=9, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_features = num_features
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #torch.Size([10])

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None] #torch.Size([200,1])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None] #torch.Size([200,1])
        eps = torch.randn_like(x) #torch.Size([200,9]) --> 9 is the nr of table columns
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps # torch.Size([200, 9])

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,)) # torch.Size([n])

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.num_features)).to(self.device) #float32
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None]  #torch.Size([200, 1])
                alpha_hat = self.alpha_hat[t][:, None]
                beta = self.beta[t][:, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 255).type(torch.uint8)
        x = [[float(t) for t in sublist] for sublist in x]

        save_synthetic_data(x)

        return x


def train(args):
    device = args.device
    dataloader = get_data(args)
    model = MLPDiffusion(d_in=args.num_features, d_layers=args.d_layers, dropouts=args.dropouts).to(device)
    # model = LSTMDiffusion(input_size=args.num_features, hidden_size=64, num_layers=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(num_features=args.num_features, device=device)
    # l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, images in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            # print("Data shape:", images.shape)
            # print("Timestep shape:", t.shape)
            x_t, noise = diffusion.noise_images(images, t)

            # Convert from float64 to float32
            x_t = x_t.to(torch.float32)
            noise = noise.to(torch.float32)

            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            # logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

    sampled_data = diffusion.sample(model, n=996)

    # Save data
    save_synthetic_data(sampled_data)

    torch.save(model.state_dict(),  f"ckpt.pt")


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "Diffusion"
    args.epochs = 1000
    args.batch_size = 10
    args.num_features = 9
    args.d_layers = [128, 128, 128]
    args.dropouts = 0.3
    args.dataset_path = os.path.join(Path(__file__).parents[1], 'data', 'totalturnover/')
    args.device = "cuda" #cuda
    args.lr = 0.003
    train(args)


if __name__ == '__main__':
    launch()

    # Sampling from saved model
    # device = "cpu"
    # num_features = 9
    # n = 100
    #
    # #Load model
    # model = MLPDiffusion(d_in=num_features, d_layers=[256, 256], dropouts=0.2).to(device)
    # ckpt = torch.load("ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(device=device)
    #
    # #Sample
    # x = diffusion.sample(model, n)

