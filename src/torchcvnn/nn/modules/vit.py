# MIT License

# Copyright (c) 2024 Jeremy Fix

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Standard imports
from typing import Callable

# External imports
import torch
import torch.nn as nn

# Local imports
from .normalization import LayerNorm
from .activation import modReLU, MultiheadAttention
from .dropout import Dropout


class ViTLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = LayerNorm,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):

        super(ViTLayer, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm1 = norm_layer(hidden_dim, **factory_kwargs)
        self.attn = MultiheadAttention(
            embed_dim=hidden_dim,
            dropout=attention_dropout,
            num_heads=num_heads,
            batch_first=True,
            **factory_kwargs
        )
        self.dropout = Dropout(dropout)
        self.norm2 = norm_layer(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim, **factory_kwargs),
            norm_layer(mlp_dim),
            modReLU(),
            Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim, **factory_kwargs),
            Dropout(dropout),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        x = x + self.dropout(self.attn(norm_x, norm_x, norm_x, need_weights=False)[0])
        x = x + self.ffn(self.norm2(x))

        return x


class ViT(nn.Module):

    def __init__(
        self,
        patch_embedder: nn.Module,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        norm_layer: Callable[..., nn.Module] = LayerNorm,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):
        super(ViT, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.patch_embedder = patch_embedder
        self.dropout = Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                ViTLayer(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    norm_layer=norm_layer,
                    **factory_kwargs
                )
            )
        self.layers = nn.Sequential(*self.layers)
        self.norm = norm_layer(hidden_dim, **factory_kwargs)

    def forward(self, x):
        # x : (B, C, H, W)
        embedding = self.patch_embedder(x)  # (B, embed_dim, num_patch_H, num_patch_W)

        # Transpose to (B, "seq_len"=num_patches, embed_dim)
        embedding = embedding.flatten(2).transpose(1, 2).contiguous()

        out = self.layers(self.dropout(embedding))

        out = self.norm(out)

        # Transpose to batch_first
        return out