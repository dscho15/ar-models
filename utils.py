import torch
from timm.models.vision_transformer import Block

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


class MHSA(torch.nn.Module):

    # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

    def __init__(self, dim: int = 128, num_heads: int = 4, mlp_ratio: int = 2):
        super(MHSA, self).__init__()
        self.block = Block(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.block(x)


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
        
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        # rescale_timesteps=rescale_timesteps,
    )
