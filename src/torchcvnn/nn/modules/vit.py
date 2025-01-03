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

# External imports
import torch
import torch.nn as nn

# Local imports
import torchcvnn.nn as c_nn


class ViTLayer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        embed_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):

        super(ViTLayer, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.norm1 = c_nn.LayerNorm(embed_dim, **factory_kwargs)
        self.attn = c_nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, **factory_kwargs
        )
        self.norm2 = c_nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, **factory_kwargs),
            c_nn.modReLU(),
            nn.Linear(4 * embed_dim, embed_dim, **factory_kwargs),
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        x = x + self.attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))

        return x


class ViT(nn.Module):

    def __init__(
        self,
        patch_embedder: nn.Module,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.complex64,
    ):
        super(ViT, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.patch_embedder = patch_embedder
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(ViTLayer(num_heads, embed_dim, **factory_kwargs))

    def forward(self, x):
        # x : (B, C, H, W)
        embedding = self.patch_embedder(x)  # (B, embed_dim, num_patch_H, num_patch_W)

        # Transpose to (B, "seq_len"=num_patches, embed_dim)
        embedding = embedding.flatten(2).transpose(1, 2)

        in_vit_layer = embedding
        for layer in self.layers:
            in_vit_layer = layer(in_vit_layer)

        # Transpose to batch_first
        return in_vit_layer
