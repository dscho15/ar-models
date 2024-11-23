import torch
from timm.models.vision_transformer import Block


class MHSA(torch.nn.Module):
    # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: int = 2):
        super(MHSA, self).__init__()
        self.block = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)
