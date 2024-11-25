import torch
import torch.nn as nn
import math

from torch.utils.checkpoint import checkpoint
from losses.respace import SpacedDiffusion, space_timesteps
import gaussian_diffusion as gd


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
) -> SpacedDiffusion:
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
    )


class DiffLoss(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth,
        width,
        num_sampling_steps,
        grad_checkpointing=False,
    ):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels

        self.train_diffusion = create_diffusion(
            timestep_respacing="", noise_schedule="cosine"
        )
        self.gen_diffusion = create_diffusion(
            timestep_respacing=num_sampling_steps, noise_schedule="cosine"
        )

    def forward(self, target: torch.FloatTensor, z: torch.FloatTensor, mask: torch.BoolTensor = None):

        t = torch.randint(
            0,
            self.train_diffusion.num_timesteps,
            (target.shape[0],),
            device=target.device,
        )

        model_kwargs = dict(c=z)

        loss_dict = self.train_diffusion.training_losses(
            self.net, target, t, model_kwargs
        )

        loss = loss_dict["loss"]

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()

        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):

        # diffusion loss sampling

        if not cfg == 1.0:

            noise = torch.randn(z.shape[0] // 2, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg

        else:

            noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn,
            noise.shape,
            noise,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            temperature=temperature,
        )

        return sampled_token_latent
