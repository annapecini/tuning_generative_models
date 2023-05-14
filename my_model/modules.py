import torch
from torch import nn
from torch import Tensor
from typing import Type, List, Union
import math
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    '''
    Generates sinusoidal positional embedding tensor. In this case, position corresponds to time. For more information
        on sinusoidal embeddings, see ["Positional Encoding - Additional Details"](https://www.assemblyai.com/blog/how-imagen-actually-works/#timestep-conditioning).
    '''

    def __init__(self, dim: int):
        """
        :param dim: Dimensionality of the embedding space
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of positions (i.e. times) to generate embeddings for.
        :return: T x D tensor where T is the number of positions/times and D is the dimensionality of the embedding
            space
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class MLP(nn.Module):
    class Block(nn.Module):
        """The main building block of `MLP`.
        Block: (in) -> Linear -> Activation -> Dropout -> (out)
        """

        def __init__(
                self,
                d_in: int,
                d_out: int,
                bias: bool,
                dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    def __init__(
            self,
            d_in: int,
            d_layers: List[int],
            dropouts: Union[float, List[float]],
            d_out: int,
    ) -> None:
        super().__init__()
        if isinstance(dropouts, float):
            dropouts = [dropouts] * len(d_layers)
        assert len(d_layers) == len(dropouts)

        self.blocks = nn.ModuleList(
                    [
                        MLP.Block(
                            d_in=d_layers[i - 1] if i else d_in,
                            d_out=d,
                            bias=True,
                            dropout=dropout,
                        )
                        for i, (d, dropout) in enumerate(zip(d_layers, dropouts))
                    ]
                )
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    @classmethod
    def make_baseline(
            cls: Type['MLP'],
            d_in: int,
            d_layers: List[int],
            dropout: float,
            d_out: int,
    ) -> 'MLP':
        assert isinstance(dropout, float)
        if len(d_layers) > 2:
            assert len(set(d_layers[1:-1])) == 1, (
                'if d_layers contains more than two elements, then'
                ' all elements except for the first and the last ones must be equal.'
            )
        return MLP(
            d_in=d_in,
            d_layers=d_layers,  # type: ignore
            dropouts=dropout,
            d_out=d_out,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.float()
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x


class MLPDiffusion(nn.Module):
    def __init__(self, d_in, d_layers, dropouts, dim_t=128):
        super().__init__()
        self.dim_t = dim_t

        self.mlp = MLP.make_baseline(dim_t, d_layers, dropouts, d_in)

        self.proj = nn.Linear(d_in, dim_t)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim_t),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

    def forward(self, x, timesteps):
        emb = self.time_embed(timesteps)
        x = self.proj(x) + emb
        return self.mlp(x)


class LSTMDiffusion(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dim_t=128):
        super(LSTMDiffusion, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dim_t = dim_t

        self.proj = nn.Linear(self.input_size, self.dim_t)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim_t),
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.LSTMBlocks = nn.ModuleList(
            [
                nn.LSTMCell(
                    input_size=self.hidden_size if i else self.dim_t,
                    hidden_size=self.hidden_size,
                    bias=self.bias
                )
                for i in range(num_layers)
            ]
        )
        self.linear = nn.Linear(self.hidden_size if num_layers else self.dim_t, self.input_size)

    def forward(self, x, timesteps):
        outputs, num_samples = [], x.size()[0]
        h_t = torch.zeros(num_samples, self.dim_t, dtype=torch.float32)
        c_t = torch.zeros(num_samples, self.dim_t, dtype=torch.float32)

        emb = self.time_embed(timesteps)
        x = self.proj(x) + emb

        for i in range(num_samples):
            for block in self.LSTMBlocks:
                print(h_t.shape)
                print(x[i].shape)
                h_t, c_t = block(x[i], (h_t, c_t))

            output = self.linear(h_t)  # output from the last FC layer
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs
