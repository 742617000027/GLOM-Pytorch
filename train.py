import torch
from tqdm import tqdm

from model import GLOM

if __name__ == '__main__':

    model = GLOM(n_cycles=10,
                 n_levels=5,
                 in_channels=1,
                 embedding_dims=10,
                 patch_size=(4, 4))

    for _ in tqdm(range(100)):
        data = torch.rand(1, 1, 16, 16)
        out = model(data)
