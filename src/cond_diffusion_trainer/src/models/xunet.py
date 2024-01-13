import numpy as np
import torch
from einops import rearrange


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def out_init_scale():
    raise NotImplementedError


def nearest_neighbor_upsample(h):
    B, F, C, H, W = h.shape
    h = torch.nn.functional.interpolate(h, scale_factor=(1, 2, 2), mode='nearest')
    return h.view(B, F, C, 2 * H, 2 * W)


def avgpool_downsample(h, k=2):
    B, F, C, H, W = h.shape
    h = h.view(B * F, C, H, W)
    h = torch.nn.functional.avg_pool2d(h, kernel_size=k, stride=k)
    return h.view(B, F, C, H // 2, W // 2)


#   return nn.avg_pool(h, (1, k, k), (1, k, k))
# raise NotImplementedError

def posenc_ddpm(timesteps, emb_ch: int, max_time=1000.):
    """Positional encodings for noise levels, following DDPM."""
    # 1000 is the magic number from DDPM. With different timesteps, we
    # normalize by the number of steps but still multiply by 1000.
    timesteps = timesteps.float()
    timesteps *= (1000. / max_time)
    half_dim = emb_ch // 2
    # 10000 is the magic number from transformers.
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim) * -emb)
    emb = emb.reshape(*([1] * (timesteps.ndim - 1)), emb.shape[-1])
    emb = timesteps[..., None] * emb
    emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim=-1).float()

    return emb


class GroupNorm(torch.nn.Module):
    """Group normalization, applied over frames."""

    def __init__(self, num_groups=32, num_channels=64):
        super().__init__()
        self.gn = torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

    def forward(self, h):
        B, _, C, H, W = h.shape
        h = self.gn(h.reshape(B * 2, C, H, W))
        return h.reshape(B, 2, C, H, W)


class FiLM(torch.nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, features: int, emb_ch=1024):
        super().__init__()
        self.features = features
        self.dense = torch.nn.Linear(emb_ch, 2 * features)

    def forward(self, h, emb):
        emb = self.dense(torch.nn.functional.silu(emb.transpose(-1, -3))).transpose(-1, -3)
        scale, shift = torch.split(emb, self.features, dim=-3)

        return h * (1. + scale) + shift


class ResnetBlock(torch.nn.Module):
    """BigGAN-style residual block, applied over frames."""

    def __init__(self,
                 in_features, out_features: int = None,
                 dropout: float = 0,
                 resample: str = None):
        super().__init__()
        self.in_features = in_features
        self.features = out_features
        self.dropout = dropout
        self.resample = resample

        if resample is not None:
            self.updown = {
                'up': nearest_neighbor_upsample,
                'down': avgpool_downsample,
            }[resample]
        else:
            self.updown = torch.nn.Identity()

        self.groupnorm0 = GroupNorm(num_channels=in_features)
        self.groupnorm1 = GroupNorm(num_channels=self.features)

        self.conv1 = torch.nn.Conv2d(in_channels=in_features,
                                     out_channels=self.features,
                                     kernel_size=3,
                                     stride=1,
                                     padding='same')
        self.film = FiLM(self.features)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(in_channels=self.features,
                                     out_channels=self.features,
                                     kernel_size=3,
                                     padding='same',
                                     stride=1)

        if in_features != out_features:
            self.dense = torch.nn.Conv2d(in_features, out_features, kernel_size=1)

        torch.nn.init.zeros_(self.conv2.weight)

    def forward(self, h_in, emb):

        B, F, C, H, W = h_in.shape

        assert C == self.in_features

        h = torch.nn.functional.silu(self.groupnorm0(h_in))
        h = self.conv1(h.view(B * F, C, H, W))
        h = h.view(B, F, self.features, H, W)
        h = self.groupnorm1(h)
        h = self.film(h, emb)
        h = self.dropout(h)
        h = self.conv2(h.view(B * F, self.features, H, W)).view(B, F, self.features, H, W)

        if C != self.features:
            h_in = self.dense(rearrange(h_in, 'b f c h w -> (b f) c h w'))
            h_in = rearrange(h_in, '(b f) c h w -> b f c h w', b=B)

        return self.updown((h + h_in) / np.sqrt(2))


class AttnLayer(torch.nn.Module):

    def __init__(self, attn_heads=4, in_channels=32):
        super().__init__()

        self.in_channels = in_channels
        self.attn_heads = attn_heads
        self.attn = torch.nn.MultiheadAttention(in_channels, attn_heads, batch_first=True)
        # hidden_dim = attn_heads * in_channels
        # self.q = torch.nn.Conv1d(in_channels, hidden_dim, 1)
        # self.kv = torch.nn.Conv1d(in_channels, hidden_dim*2, 1)

    def forward(self, q, kv):
        assert len(q.shape) == 3, "make sure the size if [B, C, H*W]"
        assert len(kv.shape) == 3, "make sure the size if [B, C, H*W]"
        assert q.shape[1] == self.in_channels

        q = rearrange(q, "b c l -> b l c")
        kv = rearrange(kv, "b c l -> b l c")

        out = self.attn(q, kv, kv)[0]

        return rearrange(out, "b l c -> b c l")


class AttnBlock(torch.nn.Module):

    def __init__(self, attn_type, attn_heads=4, in_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.attn_type = attn_type
        self.attn_heads = attn_heads

        self.groupnorm = GroupNorm(num_channels=in_channels)
        self.attn_layer = AttnLayer(attn_heads=attn_heads, in_channels=in_channels)
        # self.attn_layer1 = AttnLayer(attn_heads=attn_heads, in_channels=in_channels)
        self.linear = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, h_in):
        B, F, C, H, W = h_in.shape

        assert self.in_channels == C, f"{self.in_channels} {C}"

        h = self.groupnorm(h_in)
        h0 = h[:, 0].reshape(B, C, H * W)
        h1 = h[:, 1].reshape(B, C, H * W)

        if self.attn_type == 'self':
            h0 = self.attn_layer(q=h0, kv=h0)
            h1 = self.attn_layer(q=h1, kv=h1)

        elif self.attn_type == 'cross':
            h_0 = self.attn_layer(q=h0, kv=h1)
            h_1 = self.attn_layer(q=h1, kv=h0)

            h0 = h_0
            h1 = h_1
        else:
            raise NotImplementedError(self.attn_type)

        h = torch.stack([h0, h1], axis=1)
        h = h.view(B * F, C, H, W)
        h = self.linear(h)
        h = h.view(B, F, C, H, W)

        return (h + h_in) / np.sqrt(2)


class XUNetBlock(torch.nn.Module):

    def __init__(self, in_channels, features, use_attn=False, attn_heads=4, dropout=0):

        super().__init__()

        self.in_channels = in_channels
        self.features = features
        self.use_attn = use_attn
        self.attn_heads = attn_heads
        self.dropout = dropout

        self.resnetblock = ResnetBlock(in_features=in_channels,
                                       out_features=features,
                                       dropout=dropout)

        if use_attn:
            self.attnblock_self = AttnBlock(attn_type="self",
                                            attn_heads=attn_heads,
                                            in_channels=features)
            self.attnblock_cross = AttnBlock(attn_type="cross",
                                             attn_heads=attn_heads,
                                             in_channels=features)

    def forward(self, x, emb):

        assert x.shape[-3] == self.in_channels, f"check if channel size is correct, {x.shape[-3]}!={self.in_channels}"

        h = self.resnetblock(x, emb)

        if self.use_attn:
            h = self.attnblock_self(h)
            h = self.attnblock_cross(h)

        return h


class ConditioningProcessor(torch.nn.Module):

    def __init__(self, emb_ch, H, W,
                 num_resolutions):

        super().__init__()

        self.emb_ch = emb_ch
        self.num_resolutions = num_resolutions

        self.logsnr_emb_emb = torch.nn.Sequential(
            torch.nn.Linear(emb_ch, emb_ch),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_ch, emb_ch)
        )

        D = 144

        convs = []
        for i_level in range(self.num_resolutions):
            convs.append(torch.nn.Conv2d(in_channels=D,
                                         out_channels=self.emb_ch,
                                         kernel_size=3,
                                         stride=2 ** i_level, padding=1))

        self.convs = torch.nn.ModuleList(convs)

    def forward(self, batch):

        B, C, H, W = batch['x'].shape

        logsnr = torch.clip(batch['logsnr'], -20, 20)
        logsnr = 2 * torch.arctan(torch.exp(-logsnr / 2)) / torch.pi
        logsnr_emb = posenc_ddpm(logsnr, emb_ch=self.emb_ch, max_time=1.)
        logsnr_emb = self.logsnr_emb_emb(logsnr_emb)

        return logsnr_emb


class XUNet(torch.nn.Module):
    H: int = 128
    W: int = 128
    ch: int = 256
    ch_mult: tuple[int] = (1, 2, 2, 4)
    emb_ch: int = 1024
    num_res_blocks: int = 3
    attn_resolutions: tuple[int] = (2, 3, 4)  # actually, level of depth, from 0 th N
    attn_heads: int = 4
    dropout: float = 0.1

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()

        assert self.H % (2 ** (
                    len(self.ch_mult) - 1)) == 0, f"Size of the image must me multiple of {2 ** (len(self.ch_mult) - 1)}"
        assert self.W % (2 ** (
                    len(self.ch_mult) - 1)) == 0, f"Size of the image must me multiple of {2 ** (len(self.ch_mult) - 1)}"

        self.num_resolutions = len(self.ch_mult)
        self.conditioningprocessor = ConditioningProcessor(
            emb_ch=self.emb_ch,
            num_resolutions=self.num_resolutions,
            H=self.H,
            W=self.W
        )

        self.conv = torch.nn.Conv2d(3, self.ch, kernel_size=3, stride=1, padding='same')

        # channel size
        self.dim_in = [self.ch] + (self.ch * np.array(self.ch_mult)[:-1]).tolist()
        self.dim_out = (self.ch * np.array(self.ch_mult)).tolist()

        # upsampling
        self.xunetblocks = torch.nn.ModuleList([])
        for i_level in range(self.num_resolutions):

            single_level = torch.nn.ModuleList([])
            for i_block in range(self.num_res_blocks):
                use_attn = i_level in self.attn_resolutions

                single_level.append(
                    XUNetBlock(
                        in_channels=self.dim_in[i_level] if i_block == 0 else self.dim_out[i_level],
                        features=self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn,
                    )
                )

            if i_level != self.num_resolutions - 1:
                single_level.append(ResnetBlock(in_features=self.dim_out[i_level],
                                                out_features=self.dim_out[i_level],
                                                dropout=self.dropout,
                                                resample='down'))
            self.xunetblocks.append(single_level)

        # middle

        self.middle = XUNetBlock(
            in_channels=self.dim_out[-1],
            features=self.dim_out[-1],
            dropout=self.dropout,
            attn_heads=self.attn_heads,
            use_attn=self.num_resolutions in self.attn_resolutions)

        # upsample
        self.upsample = torch.nn.ModuleDict()
        for i_level in reversed(range(self.num_resolutions)):
            single_level = torch.nn.ModuleList([])
            use_attn = i_level in self.attn_resolutions

            for i_block in range(self.num_res_blocks + 1):

                if i_block == 0:
                    # then the input size is same as output of previous level

                    prev_h_channels = self.dim_out[i_level + 1] if (i_level + 1 < len(self.dim_out)) else self.dim_out[
                        i_level]
                    prev_emb_channels = self.dim_out[i_level]

                elif i_block == self.num_res_blocks:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_in[i_level]

                else:
                    prev_h_channels = self.dim_out[i_level]
                    prev_emb_channels = self.dim_out[i_level]

                    # self.dim_out[i_level]*2 if i_block != self.num_res_blocks else (self.dim_out[i_level] + self.dim_in[i_level])

                in_channels = prev_h_channels + prev_emb_channels

                single_level.append(
                    XUNetBlock(
                        in_channels=in_channels,
                        features=self.dim_out[i_level],
                        dropout=self.dropout,
                        attn_heads=self.attn_heads,
                        use_attn=use_attn)
                )

            if i_level != 0:
                single_level.append(
                    ResnetBlock(in_features=self.dim_out[i_level],
                                out_features=self.dim_out[i_level],
                                dropout=self.dropout, resample='up')
                )
            self.upsample[str(i_level)] = single_level

        self.lastgn = GroupNorm(num_channels=self.ch)
        self.lastconv = torch.nn.Conv2d(in_channels=self.ch, out_channels=3, kernel_size=3, stride=1, padding='same')
        torch.nn.init.zeros_(self.lastconv.weight)

    def forward(self, batch, *, cond_mask):

        B, C, H, W = batch['x'].shape

        for key, temp in batch.items():
            assert temp.shape[0] == B, f"{key} should have batch size of {B}, not {temp.shape[0]}"
        assert B == cond_mask.shape[0]
        assert (H, W) == (self.H, self.W), ((H, W), (self.H, self.W))

        logsnr_emb = self.conditioningprocessor(batch)

        h = torch.stack([batch['x'], batch['z']], dim=1)
        h = self.conv(rearrange(h, 'b f c h w -> (b f) c h w'))
        h = rearrange(h, '(b f) c h w -> b f c h w', b=B, f=2)

        # downsampling
        hs = [h]
        for i_level in range(self.num_resolutions):

            emb = logsnr_emb[..., None, None]

            for i_block in range(self.num_res_blocks):
                h = self.xunetblocks[i_level][i_block](h, emb)
                hs.append(h)

            if i_level != self.num_resolutions - 1:
                h = self.xunetblocks[i_level][-1](
                    h, emb)
                hs.append(h)

        # middle, 1x block
        emb = logsnr_emb[..., None, None]

        h = self.middle(h, emb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            emb = logsnr_emb[..., None, None]

            for i_block in range(self.num_res_blocks + 1):
                h = torch.concat([h, hs.pop()], dim=-3)

                orishape = h.shape
                h = self.upsample[str(i_level)][i_block](h, emb)

            if i_level != 0:
                h = self.upsample[str(i_level)][-1](h, emb)

        assert not hs  # check hs is empty

        h = torch.nn.functional.silu(self.lastgn(h))  # [B, F, self.ch, 128, 128]
        return rearrange(self.lastconv(rearrange(h, 'b f c h w -> (b f) c h w')), '(b f) c h w -> b f c h w', b=B)[:, 1]


if __name__ == "__main__":
    h, w = 56, 56
    b = 8
    a = XUNet(H=h, W=w, ch=128)

    batch = {
        'x': torch.zeros(b, 3, h, w).to(device),
        'z': torch.zeros(b, 3, h, w).to(device),
        'logsnr': torch.tensor([10] * (2 * b)).reshape(b, 2),
    }

    print(a(batch, cond_mask=torch.tensor([True] * b).to(device)).shape)