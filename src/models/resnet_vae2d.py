import torch
from torch import Tensor, nn
from torch.nn import functional as F

from math import ceil
from typing import List

from .base import BaseVAE
from .encoder_wrapper import EncoderWrapper
from .inception import InceptionV3
from .resnet2d import BasicBlock2d, TransposeBasicBlock2d
from .upscale2d import Upscale2d
from .util import get_activation, get_pooling2d


class ResNetVAE2d(BaseVAE):

    def __init__(self,
                 name: str,
                 latent_dim: int,
                 hidden_dims: List[int],
                 width: int,
                 height: int,
                 channels: int,
                 dropout: float = 0.4,
                 output_activation: str = 'sigmoid',
                 pooling: str = None,
                 enable_fid: bool = False,
                 fid_blocks: List[int] = [3],
                 progressive_growing: int = 0) -> None:
        super(ResNetVAE2d, self).__init__(name=name, latent_dim=latent_dim)
        self.width = width
        self.height = height
        self.channels = channels
        self.hidden_dims = hidden_dims.copy()

        self.enable_fid = enable_fid
        if enable_fid:
            self.inception = InceptionV3(fid_blocks, use_fid_inception=True)

        if pooling is not None:
            pool_fn = get_pooling2d(pooling)

        # Encoder
        modules = []
        in_features = channels
        for h_dim in hidden_dims:
            modules.append(BasicBlock2d(in_features, h_dim))
            if pooling is not None:
                modules.append(pool_fn(2))
            in_features = h_dim
        layers = nn.Sequential(
            *modules,
            nn.Flatten(),
            nn.Dropout(p=dropout),
        )

        in_features = hidden_dims[-1] * width * height
        if pooling is not None:
            in_features /= 4**len(hidden_dims)
            if abs(in_features - ceil(in_features)) > 0:
                raise ValueError(
                    'noninteger number of features - perhaps there is too much pooling?'
                )
            in_features = int(in_features)
        self.encoder = EncoderWrapper(
            latent_dim=latent_dim,
            layers=layers,
            mu=nn.Sequential(
                nn.Linear(in_features, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            ),
            var=nn.Sequential(
                nn.Linear(in_features, latent_dim),
                nn.BatchNorm1d(latent_dim),
                nn.ReLU(),
            ),
        )

        # Decoder
        hidden_dims.reverse()
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        modules = []
        sandwich_layers = []
        in_features = hidden_dims[0]
        for h_dim in hidden_dims:
            layer = TransposeBasicBlock2d(in_features, h_dim)
            modules.append(layer)
            sandwich_layers.append((layer, h_dim))
            in_features = h_dim
        self.sandwich_layers = sandwich_layers
        self.decoder = nn.Sequential(*modules)
        self.progressive_growing = progressive_growing
        if progressive_growing != 0:
            if width != height:
                raise ValueError(
                    f'Progressive growing is only supported for square images')
            res = width // 2**(progressive_growing - 1)
            output_layers = []
            for i in range(progressive_growing):
                out_features = res * res * channels
                layer = nn.Conv2d(in_features,
                                  out_features,
                                  kernel_size=3,
                                  padding=1)
                output_layers.append(layer)
                in_features = out_features
                res *= 2
            self.decoder_output = nn.ModuleList(output_layers)
        else:
            self.decoder_output = nn.Sequential(
                nn.Conv2d(in_features,
                          width * height * channels // 4,
                          kernel_size=3,
                          padding=1))
        self.decoder_activation = get_activation(output_activation)

    def get_sandwich_layers(self) -> List[nn.Module]:
        return self.sandwich_layers

    def get_encoder(self) -> List[nn.Module]:
        return self.encoder

    def encode(self, input: Tensor) -> List[Tensor]:
        if input.shape[-3:] != (self.channels, self.height, self.width):
            raise ValueError('wrong input shape')
        return self.encoder(input)

    def decode(self,
               z: Tensor,
               lod: int = 0,
               alpha: float = 1.0,
               **kwargs) -> Tensor:
        x = self.decoder_input(z)
        x = x.view(x.shape[0], self.hidden_dims[-1], 2, 2)
        x = self.decoder(x)
        if self.progressive_growing != 0:
            if lod == 0:
                # Full output resolution, no layer mixing
                for layer in self.decoder_output:
                    x = layer(x)
                x = F.max_pool2d(x, 2)
                x = x.view(x.shape[0], self.channels, self.height, self.width)
            else:
                # Stop at the appropriate output resolution
                # 3 - 0 = 3     28x28
                # 3 - 1 = 2     14x14
                # 3 - 2 = 1     7x7
                layer_i = len(self.decoder_output) - lod
                width = self.width // 2**lod
                height = self.height // 2**lod
                for i in range(layer_i):
                    if i == layer_i - 1:
                        # Lower output layer
                        x = self.decoder_output[i + 0](x)
                        a = F.max_pool2d(x, 2)
                        a = a.view(a.shape[0], self.channels, height, width)
                        a = Upscale2d()(a)

                        # Upper output layer
                        b = self.decoder_output[i + 1](x)
                        b = F.max_pool2d(b, 2)
                        b = b.view(b.shape[0], self.channels, height * 2,
                                   width * 2)

                        # Compute interpolated output
                        x = a * (1.0 - alpha) + b * alpha
                    else:
                        x = self.decoder_output[i](x)

        else:
            x = self.decoder_output(x)
            x = x.view(x.shape[0], self.channels, self.height, self.width)
        x = self.decoder_activation(x)
        return x

    def loss_function(self,
                      recons: Tensor,
                      orig: Tensor,
                      *args,
                      fid_weight: float = 1.0,
                      **kwargs) -> dict:
        if 'lod' in kwargs:
            raise NotImplementedError
            n = kwargs['lod'] - 1
            for _ in range(n):
                orig = F.max_pool2d(orig, 2)

        result = super(ResNetVAE2d,
                       self).loss_function(recons, orig, *args, **kwargs)

        if self.enable_fid:
            fid_loss = self.fid(orig, recons).sum()
            result['FID_Loss'] = fid_loss
            result['loss'] += fid_loss * fid_weight

        return result

    def fid(self, a: Tensor, b: Tensor) -> Tensor:
        if a.shape[1] == 1:
            # Convert monochrome to RGB
            a = a.repeat(1, 3, 1, 1)
            b = b.repeat(1, 3, 1, 1)
        a = self.inception(a)
        b = self.inception(b)
        fid = [torch.mean((x - y)**2) for x, y in zip(a, b)]
        fid = torch.Tensor(fid)
        return fid
