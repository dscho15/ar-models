from models.vae import AutoencoderKL
import torch

# create model
vae = AutoencoderKL(
    embed_dim=16, ch_mult=(1, 1, 2, 2, 4), ckpt_path="checkpoints/model.ckpt"
)

for param in vae.parameters():
    param.requires_grad = False
