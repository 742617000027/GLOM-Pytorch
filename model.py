import numpy as np
import torch
import torch.nn as nn


class BottomUp(nn.Module):
    def __init__(self):
        super(BottomUp, self).__init__()


class TopDown(nn.Module):
    def __init__(self):
        super(TopDown, self).__init__()


class GLOM(nn.Module):
    def __init__(self, n_cycles, n_levels, in_channels, embedding_dims, patch_size, L=10, beta=1e-3):
        super(GLOM, self).__init__()
        self.n_cycles = n_cycles
        self.n_levels = n_levels
        self.embedding_dims = embedding_dims
        self.patch_size = patch_size
        self.L = L
        self.beta = beta
        self.init_embed = nn.Conv2d(in_channels=in_channels,
                                    out_channels=embedding_dims,
                                    kernel_size=patch_size,
                                    padding=(0, 0),
                                    stride=patch_size)
        self.bottom_up = nn.ModuleList()
        self.top_down = nn.ModuleList()
        self.BN = nn.ModuleList()
        for i in range(n_levels):
            self.bottom_up.append(nn.Linear(embedding_dims + 4 * L, embedding_dims))
            self.top_down.append(nn.Linear(embedding_dims + 4 * L, embedding_dims))
            self.BN.append(nn.BatchNorm2d(embedding_dims))
        self.w1 = nn.Parameter(torch.tensor(1.))
        self.w2 = nn.Parameter(torch.tensor(1.))
        self.w3 = nn.Parameter(torch.tensor(1.))
        self.w4 = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        N, _, H, W = x.shape
        H = H // self.patch_size[0]
        W = W // self.patch_size[1]

        embedding = torch.stack(
            [self.init_embed(x)] + [torch.zeros((N, self.embedding_dims, H, W)).to(x.device)] * (self.n_levels - 1),
            dim=4)
        prev_embedding = embedding.clone()

        for t in range(self.n_cycles):
            for l in range(self.n_levels):
                prev_embedding[:, :, :, :, l] = self.BN[l](prev_embedding[:, :, :, :, l])
                for h in range(H):
                    for w in range(W):
                        loc = torch.stack([torch.cat([self.location(2 * h / H - 1), self.location(2 * w / W - 1)])] * N)
                        bottom_up = \
                            self.bottom_up[l](torch.cat([prev_embedding[:, :, h, w, l - 1], loc], dim=1)) \
                                if l > 0 else 0.
                        top_down = \
                            self.top_down[l](torch.cat([prev_embedding[:, :, h, w, l + 1], loc], dim=1)) \
                                if l < self.n_levels - 1 else 0.
                        attention_weighted_average = \
                            torch.sum(prev_embedding[:, :, :, :, l] *
                                      self.attention(prev_embedding[:, :, :, :, l], h, w), dim=(2, 3)) \
                                if torch.sum(prev_embedding[:, :, :, :, l]) != 0. else 0.
                        embedding[:, :, h, w, l] = \
                            self.w1 * prev_embedding[:, :, h, w, l] + \
                            self.w2 * bottom_up + \
                            self.w3 * top_down + \
                            self.w4 * attention_weighted_average
            prev_embedding = embedding.clone()

        return embedding

    def attention(self, embedding, h, w):
        w = torch.exp(self.beta * torch.einsum('be,behw->bhw', embedding[:, :, h, w], embedding))
        return w / torch.sum(w)

    def location(self, p):
        '''
         Using (4) from https://arxiv.org/abs/2003.08934 as location embedding.
        '''
        location_embeddings = []
        for l in range(self.L):
            location_embeddings.extend(torch.tensor([np.sin(2 ** l * np.pi * p), np.cos(2 ** l * np.pi * p)]).float())
        return torch.stack(location_embeddings)
