import torch
import math

from torch import nn


def modulate(x: torch.FloatTensor, shift: torch.FloatTensor, scale: torch.FloatTensor) -> torch.FloatTensor:
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: int, dim: int, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        
        half = dim // 2
        
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        
        args = t[:, None].float() * freqs[None]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
            
        return embedding

    def forward(self, t: torch.FloatTensor) -> torch.FloatTensor:
        
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        t_emb = self.mlp(t_freq)
        
        return t_emb
    

class ResBlock(nn.Module):

    def __init__(self, channels: int):
        
        super().__init__()
        
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x: torch.FloatTensor, y: torch.FloatTensor):
        
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        
        h = self.mlp(h)
        
        return x + gate_mlp * h


class FinalLayer(nn.Module):

    def __init__(self, model_channels: int, out_channels: int):
        
        super().__init__()
        
        self.norm_final = nn.LayerNorm(
            model_channels, elementwise_affine=False, eps=1e-6
        )
        
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift, scale)
        
        x = self.linear(x)
        
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        z_channels: int,
        num_res_blocks: int,
        grad_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.FloatTensor, t: torch.FloatTensor, c: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        for block in self.res_blocks:
            x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x: torch.FloatTensor, t: torch.FloatTensor, c: torch.FloatTensor, cfg_scale: float) -> torch.FloatTensor:

        half = x[: len(x) // 2]

        combined = torch.cat([half, half], dim=0)

        model_out = self.forward(combined, t, c)

        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        eps = torch.cat([half_eps, half_eps], dim=0)

        return torch.cat([eps, rest], dim=1)
