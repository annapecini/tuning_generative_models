import torch
import numpy as np
import os

if __name__ == '__main__':
    beta_start = 1e-4
    beta_end = 0.02
    noise_steps = 1000

    beta = torch.linspace(beta_start, beta_end, noise_steps)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    print(torch.randint(low=1, high=noise_steps, size=(5,)))


    # alpha_hat = torch.cumprod(alpha, dim=0)
    #
    # sqrt_alpha_hat = torch.sqrt(alpha_hat[0])
    # sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[0])
    #
    #
    # x = torch.from_numpy(np.load(os.path.join("/Users/apecini/Documents/thesis/master-thesis/data/totalturnover/", 'X_num_train.npy'), allow_pickle=True))
    # eps = torch.randn_like(x[0])
    # out = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
    # print(out.shape)
    # print((torch.randint(low=1, high=noise_steps, size=(3,))).shape)
