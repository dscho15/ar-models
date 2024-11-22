import timm
import torch
import einops
import numpy as np

from timm.models.vision_transformer import Block
from scipy import stats

from torchvision.io import read_image
from torchvision.utils import save_image


class MHSA(torch.nn.Module):

    def __init__(self):
        super(MHSA, self).__init__()
        self.block = Block(dim=128, num_heads=4, mlp_ratio=2)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)


class MAR(torch.nn.Module):

    def __init__(self, h: int = 64, w: int = 64, patch_size: int = 4, min_mask_ratio: float = 0.6):
        super(MAR, self).__init__()

        self.patch_size = patch_size
        self.seq_len = h * w // self.patch_size**2

        self.mask_ratio_gen = stats.truncnorm((min_mask_ratio - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

    def patchify(self, x: torch.FloatTensor) -> torch.FloatTensor:
        p = self.patch_size

        return einops.rearrange(
            x, "b c (n_h p1) (n_w p2) -> b (n_h n_w) (p1 p2 c)", p2=p, p1=p
        )

    def unpatchify(self, x: torch.FloatTensor, h: int = 64, w=64) -> torch.FloatTensor:
        p = self.patch_size

        n_h, n_w = h // p, w // p

        return einops.rearrange(
            x,
            "b (n_h n_w) (p1 p2 c) -> b c (n_h p1) (n_w p2)",
            p2=p,
            p1=p,
            n_h=n_h,
            n_w=n_w,
        )

    def sample_orders(self, bsz):
        orders = []

        for _ in range(bsz):

            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)

        orders = torch.Tensor(np.array(orders)).long() # (bsz, seq_len)
        return orders
    
    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape

        mask_rate = self.mask_ratio_gen.rvs(1)[0] # sample mask rate from truncated normal distribution

        num_masked_tokens = int(np.ceil(seq_len * mask_rate)) # number of masked tokens

        mask = torch.zeros(bsz, seq_len, device=x.device) # mask tensor
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens], src=torch.ones(bsz, seq_len, device=x.device)) # scatter ones to the first num_masked_tokens
        
        return mask

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        patches = self.patchify(x).contiguous() # (bsz, seq_len, embed_dim)
        orders = self.sample_orders(x.size(0)) # (bsz, seq_len)
        mask = self.random_masking(patches, orders) # (bsz, seq_len)

        patches[mask > 0] = 0.

        patches = self.unpatchify(patches, h=x.size(-2), w=x.size(-1)) # (bsz, c, h, w)

        return patches

model = MAR()

x = read_image("resized.jpg")[None] / 255.0

x = model(x)

# save image
save_image(x[0], "masked.png")