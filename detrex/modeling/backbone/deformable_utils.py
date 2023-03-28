import torch
import torch.nn as nn
from deformable_attention import DeformableAttention
from timm.models.layers import DropPath, Mlp, to_2tuple, trunc_normal_
from detectron2.modeling.backbone.utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,

)

import einops
from .convnext_utils import ConvNeXtBlock

class LayerNorm(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class reshapeMLP(nn.Module):
    def __init__(self, in_features, hidden_features, act_layer):
        super().__init__()
        self.mlp = Mlp(in_features=in_features, hidden_features=hidden_features, act_layer=act_layer)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        return einops.rearrange(x, 'b h w c -> b c h w')

class reshapeLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = nn.Linear(in_features, out_features)
    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.mlp(x)
        return einops.rearrange(x, 'b h w c -> b c h w')



class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x

class learnableDAT(nn.Module):
    def __init__(self, in_dim, out_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                dilation=1,
                                groups=1, bias=True)

        self.Deformable_attn = DeformableAttention(
            dim = 768,                   # feature dimensions
            dim_head = 64,               # dimension per head
            heads = 12,                   # attention heads
            dropout = 0.,                # dropout
            downsample_factor = 4,       # downsample factor (r in paper)
            offset_scale = 4,            # scale of offset, maximum offset
            offset_groups = None,        # number of offset groups, should be multiple of heads
            offset_kernel_size = 6,      # offset kernel size
        )

        # self.mlp = reshapeLinear(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio), act_layer=act_layer)
        self.mlp = reshapeLinear(out_dim, out_dim)


        self.norm1 = norm_layer(out_dim)
        self.norm2 = norm_layer(out_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


    def forward(self, x):
        x = self.proj(x)
        shortcut = x
        x = self.Deformable_attn(self.norm1(x))
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class learnableConv(nn.Module):
    def __init__(self, in_dim, out_dim, ls_init_value=1e-6, conv_mlp=True,):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                dilation=1,
                                groups=1, bias=True)
        

        self.convnext = ConvNeXtBlock(dim=out_dim, norm_layer=LayerNorm, ls_init_value=ls_init_value, conv_mlp=conv_mlp)


    def forward(self, x):
        x = self.proj(x)
        x = self.convnext(x)

        return x

class learnableWindowAttn(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                dilation=1,
                                groups=1, bias=True)
        
        self.norm1 = norm_layer(out_dim)
        self.attn = Attention(
            out_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        self.mlp = Mlp(in_features=out_dim, hidden_features=int(out_dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0, 3, 1, 2)





if __name__ == "__main__":
    x = torch.randn(3, 768, 40, 40)
    model = learnableDAT(768, 768, 12)
    # from IPython import embed; embed()
    print(model(x).shape)