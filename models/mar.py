import timm
import torch
import einops
import numpy as np

from timm.models.vision_transformer import Block
from scipy import stats

from models.simple_mlp_ada_ln import SimpleMLPAdaLN


class MAR(torch.nn.Module):

    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        patch_size: int = 16,
        min_mask_ratio: float = 0.6,
        encoder_depth: int = 4,
        encoder_embed_dim: int = 128,
        encoder_num_heads: int = 4,
        encoder_mlp_ratio: int = 2,
        encoder_qkv_bias: bool = True,
        encoder_norm_layer: torch.nn.Module = torch.nn.LayerNorm,
        encoder_proj_dropout: float = 0.0,
        encoder_attn_dropout: float = 0.0,
        
        decoder_depth: int = 4,
        decoder_embed_dim: int = 128,
        
        diffusion_embedding_dim: int = 128,
        diffusion_output_channels: int = 3,
        diffusion_num_res_blocks: int = 8,
        
        label_drop_prob: float = 0.7,
        n_classes: int = 10,
    ):
        super(MAR, self).__init__()

        self.patch_size = patch_size
        self.seq_len = int((h * w) / (self.patch_size**2))
        self.buffer_size = 4

        self.mask_ratio_gen = stats.truncnorm(
            (min_mask_ratio - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )
        self.label_drop_prob = label_drop_prob

        self.class_embeddings = torch.nn.Parameter(
            torch.randn(n_classes, encoder_embed_dim)
        )
        self.fake_latent = torch.nn.Parameter(torch.randn(1, encoder_embed_dim))

        self.encoder_proj = torch.nn.Linear(patch_size**2 * 3, encoder_embed_dim)
        self.encoder_blocks = torch.nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    encoder_mlp_ratio,
                    qkv_bias=encoder_qkv_bias,
                    norm_layer=encoder_norm_layer,
                    proj_drop=encoder_proj_dropout,
                    attn_drop=encoder_attn_dropout,
                )
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = torch.nn.LayerNorm(encoder_embed_dim)
        self.encoder_pos_embeddings = torch.nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )

        self.z_proj_ln = torch.nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        self.decoder_blocks = torch.nn.ModuleList(
            [
                Block(
                    encoder_embed_dim,
                    encoder_num_heads,
                    encoder_mlp_ratio,
                    qkv_bias=encoder_qkv_bias,
                    norm_layer=encoder_norm_layer,
                    proj_drop=encoder_proj_dropout,
                    attn_drop=encoder_attn_dropout,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_pos_embeddings = torch.nn.Parameter(
            torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim)
        )
        self.decoder_embed = torch.nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=True
        )
        self.decoder_mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, decoder_embed_dim)
        )
        self.decoder_norm = torch.nn.LayerNorm(decoder_embed_dim)

        # diffusion pos embeddings

        self.diffusion_pos_embeddings = torch.nn.Parameter(
            torch.zeros(1, self.seq_len, decoder_embed_dim)
        )
        
        self.simple_mlp_ada_ln = SimpleMLPAdaLN(
            decoder_embed_dim, 
            diffusion_output_channels,
            diffusion_output_channels,
            diffusion_embedding_dim,
            diffusion_num_res_blocks,
        )

    def initialize_weights(self):

        # parameters
        torch.nn.init.normal_(self.class_embeddings, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        torch.nn.init.normal_(self.decoder_mask_token, std=0.02)
        torch.nn.init.normal_(self.encoder_pos_embeddings, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embeddings, std=0.02)
        torch.nn.init.normal_(self.diffusion_pos_embeddings, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

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

        orders = torch.Tensor(np.array(orders)).long()  # (bsz, seq_len)
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape

        mask_rate = self.mask_ratio_gen.rvs(1)[
            0
        ]  # sample mask rate from truncated normal distribution

        num_masked_tokens = int(np.ceil(seq_len * mask_rate))  # number of masked tokens

        mask = torch.zeros(bsz, seq_len, device=x.device)  # mask tensor
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )  # scatter ones to the first num_masked_tokens

        return mask

    def forward_mae_encoder(
        self,
        x: torch.FloatTensor,
        mask: torch.BoolTensor,
        class_embedding: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # project to latent space
        x = self.encoder_proj(x)

        # add class embedding
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat(
            [torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1
        )
        # image = [buffer | image]
        
        mask_with_buffer = torch.cat(
            [torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1
        )
        # mask = [buffer | mask]

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
            class_embedding = (
                drop_latent_mask * self.fake_latent
                + (1 - drop_latent_mask) * class_embedding
            )

        x[:, : self.buffer_size] = class_embedding.unsqueeze(1)
        # x = [class_embedding | ...]

        # encoder position embedding
        x = x + self.encoder_pos_embeddings # position embedding
        x = self.z_proj_ln(x) # layer_norm

        # dropping
        inverted_mask_indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        x = x[inverted_mask_indices].reshape(bsz, -1, embed_dim)

        # apply transformer blocks
        for block in self.encoder_blocks:
            x = block(x)

        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(
        self, x: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:

        # decode masked embeddings 
        x = self.decoder_embed(x)
        
        # 
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.decoder_mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)

        x_after_pad = mask_tokens.clone()
        
        inverted_mask_indices = (1 - mask_with_buffer).nonzero(as_tuple=True)
        
        x_after_pad[inverted_mask_indices] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embeddings

        # apply Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.decoder_norm(x)

        x = x[:, self.buffer_size :]

        x = x + self.diffusion_pos_embeddings

        return x

    def forward_diffusion(
        self, x: torch.FloatTensor, mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        
        
        pass
        
        

    def forward(
        self, imgs: torch.FloatTensor, labels: torch.LongTensor
    ) -> torch.FloatTensor:

        # class embeds
        class_embeddings = self.class_embeddings[labels]

        # patchify
        x = self.patchify(imgs).contiguous()  # (bsz, seq_len, embed_dim)
        gt = x.clone().detach()

        orders = self.sample_orders(x.size(0))  # (bsz, seq_len)
        mask = self.random_masking(x, orders)  # (bsz, seq_len)

        # forward mae-encoder
        x = self.forward_mae_encoder(x, mask, class_embeddings)

        # pass onto mae-decoder
        z = self.forward_mae_decoder(x, mask)

        # pass on to diffusion-process
        z_pred = self.forward_diffusion(z, mask)

        return x
    
if __name__ == "__main__":
    
    model = MAR()
    
    x = torch.randn(1, 3, 64, 64)
    labels = torch.randint(0, 10, (1,))
    
    pred = model(x, labels)