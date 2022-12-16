import logging
import math
import fvcore.nn.weight_init as weight_init
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous

from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import DropPath, Mlp, trunc_normal_
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Attention, LayerScale

from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,

)
import math
from detrex.layers import LayerNorm
from .convnext_utils import _create_hybrid_backbone
from .fan import HybridEmbed
from IPython import embed

logger = logging.getLogger(__name__)


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class BaseConv(nn.Module):
    '''
    Down_Sample 
    '''
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        down_sample=1,
    ):
        super().__init__()
        input_dim, output_dim = in_channels, out_channels // (2 ** (down_sample - 1))
        block = []
        for idx in range(down_sample):
            stage_list = [
                nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding),
                LayerNorm(output_dim, channel_last=False),
                nn.GELU(),
            ]
            stage = nn.Sequential(*stage_list)
            input_dim = output_dim
            output_dim *= 2
            block.append(stage)
        block.append(
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.proj = nn.ModuleList(block)
    
    def forward(self, x):
        for stage in self.proj:
            x = stage(x)
        return x

class ConvStem(nn.Module):

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        down_sample=[4, 8],
        embed_dims=[128, 256, 768],
        checkpointing=False,
    ):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.down_sample = down_sample
        self.embed_dims = embed_dims
        self.checkpointing = checkpointing

        self.Block = []
        in_channel = 3
        for i, depth in enumerate(self.down_sample):
            if i == 0:
                depth = math.log2(depth)
            else:
                depth = math.log2(self.down_sample[i] // self.down_sample[i - 1])
            out_channel = self.embed_dims[i]
            self.Block.append(BaseConv(in_channel, out_channel, down_sample=int(depth)))
            self.Block = nn.ModuleList(self.Block)
            in_channel = out_channel

        self.proj = nn.Conv2d(in_channel, embed_dims[-1], kernel_size=2, stride=2)

    def forward(self, x):
        outputs = []
        for stage in self.Block:
            x = stage(x)
            outputs.append(x)
        x = self.proj(x)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        
        return outputs, x, (H, W)

class TokenMixing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        cha_sr = 1
        self.q = nn.Linear(dim, dim // cha_sr, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2 // cha_sr, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q * self.scale @ k.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class MIMConvStem(nn.Module):
    """ConvStem, from Early Convolutions Help Transformers See Better, Tete et
    al.
    https://arxiv.org/abs/2106.14881
    """

    def __init__(
        self,
        in_chans=3,
        embed_dim=768,
        depth=4,
        norm_layer=None,
        checkpointing=False,
    ):
        super().__init__()

        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        self.depth = depth
        self.checkpointing = checkpointing

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // (2 ** (depth - 1))
        for idx in range(depth):
            stage_list = [
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, output_dim, eps=1e-6),
                nn.GELU(),
            ]
            if idx == depth - 1:
                stage_list.append(nn.Conv2d(output_dim, embed_dim, kernel_size=1))
            stage = nn.Sequential(*stage_list)
            input_dim = output_dim
            output_dim *= 2
            stem.append(stage)
        self.proj = nn.ModuleList(stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        outputs = []
        for i, stage in enumerate(self.proj):
            if self.checkpointing and x.requires_grad:
                x = cp.checkpoint(stage, x)
            else:
                x = stage(x)
            if i >= 1:
                if i == (len(self.proj) - 1):
                    outputs.append(self.norm(x))
                else:
                    outputs.append(x)
        Hp, Wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        outputs.pop(-1)
        return outputs, x, (Hp, Wp)

class ViT(Backbone):

    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        pos_embed="abs_pos",
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_channel=[128, 256, 768, 768],
        out_ids=11,
        patch_embed="patch_embed",
        out_index = [0, 1, 2, 3],
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            window_block_indexes (list): Indexes for blocks using window attention.
            residual_block_indexes (list): Indexes for blocks using conv propagation.
            use_act_checkpoint (bool): If True, use activation checkpointing.
            pretrain_img_size (int): input` image size for pretraining models.
            pretrain_use_cls_token (bool): If True, pretrainig models use class token.
            out_feature (str): name of the feature from the last block.
        """
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.out_channel = out_channel

        if patch_embed == "pathch_embed":
            self.patch_embed = PatchEmbed(
                kernel_size=(patch_size, patch_size),
                stride=(patch_size, patch_size),
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        elif patch_embed == "MyConv":
            self.patch_embed = ConvStem(embed_dims=self.out_channel[:-1])
        elif patch_embed == "MIMConv":
            self.patch_embed = MIMConvStem(embed_dim=embed_dim, depth=4)
        elif patch_embed == "ConvNext":
            model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
            backbone = _create_hybrid_backbone(pretrained=False, pretrained_strict=False, **model_args)
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=2, embed_dim=embed_dim)

        if pos_embed == "abs_pos":
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.out_index = out_index
        self.out_ids = out_ids
        
        for i_layer in self.out_index:
            layer = nn.LayerNorm(self.out_channel[i_layer])
            layer_name = f'out_norm{i_layer}'
            self.add_module(layer_name, layer)

        self.add_module("learnable_downsample", nn.Conv2d(in_channels=embed_dim,
                                        out_channels=768,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        groups=1, bias=True))

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if isinstance(self.patch_embed, HybridEmbed):
            x, (Hp, Wp), outputs = self.patch_embed(x, return_feat=True)
        else:
            outputs, x, (Hp, Wp) = self.patch_embed(x)
        B = x.shape[0]
        if self.pos_embed is not None:
            pos_embed = get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (Hp, Wp)
            ).permute(0, 3, 1, 2).flatten(2).transpose(1, 2)
            x = x + pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.out_ids:
                outputs.append(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous())
        outputs.append(self.learnable_downsample(x.reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2).contiguous()))
        final_results = {}

        for i, out in enumerate(outputs):
            if i in self.out_index:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f'out_norm{i}')
                out = norm_layer(out)
                final_results[f'p{i}'] = out.permute(0, 3, 1, 2).contiguous()
        return final_results


class MIMConvViT(ViT):
    def __init__(self, out_index=[0, 1, 2, 3], out_channel = [192, 384, 768, 768]):
        pos_embed = "abs_pos"
        patch_embed = "MIMConv"
        super(MIMConvViT, self).__init__(
            out_index=out_index, pos_embed=pos_embed, patch_embed=patch_embed, out_channel=out_channel
        )


class MyConvViT(ViT):
    def __init__(self, out_index=[0, 1, 2, 3], out_channel = [128, 256, 768, 768]):
        pos_embed = "abs_pos"
        patch_embed = "MyConv"
        super(MyConvViT, self).__init__(
            out_index=out_index, pos_embed=pos_embed, patch_embed=patch_embed, out_channel=out_channel
        )

class ConvNextViT(ViT):
    def __init__(self, out_index=[0, 1, 2, 3], out_channel = [128, 256, 768, 768]):
        pos_embed = "abs_pos"
        patch_embed = "ConvNext"
        super(ConvNextViT, self).__init__(
            out_index=out_index, pos_embed=pos_embed, patch_embed=patch_embed, out_channel=out_channel
        )

if __name__ == "__main__":
    model = ConvNextViT([1, 2, 3])
    x = torch.randn(3, 3, 224, 224)
    x = model(x)
    for key, value in x.items():
        print(value.shape)