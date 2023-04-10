"""Vision Transformer (ViT) in PyTorch.
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman
"""
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import DropPath, Mlp, to_2tuple, trunc_normal_
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.utils import (
    PatchEmbed,
    add_decomposed_rel_pos,
    get_abs_pos,
    window_partition,
    window_unpartition,

)
from .convnext_utils import _create_hybrid_backbone, ConvNeXtBlock
from .learnable_downsample import learnableDAT, learnableConv, learnableWindowAttn, learnableCommonConv
        

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=2, feature_size=None, in_chans=3, embed_dim=384):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone.forward_features(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.post_norm = LayerNorm(embed_dim)


    def forward(self, x, return_feat=False):
        x, out_list = self.backbone.forward_features(x, return_feat=return_feat)
        B, C, H, W = x.shape
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        x = self.post_norm(x).permute(0, 2, 3, 1)
        if return_feat:
            return x , (H//self.patch_size[0], W//self.patch_size[1]), out_list
        else:
            return x , (H//self.patch_size[0], W//self.patch_size[1])





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



class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        norm=LayerNorm,
        act_layer=nn.GELU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = norm(bottleneck_channels)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(bottleneck_channels,
                               bottleneck_channels,
                               3,
                               padding=1,
                               bias=False,)
        self.norm2 = norm(bottleneck_channels)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = norm(out_channels)

        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()

    def forward(self, x):
        out = x
        for layer in [self.conv1, self.norm1, self.act1,
                      self.conv2, self.norm2, self.act2,
                      self.conv3, self.norm3]:
            x = layer(x)

        out = x + out
        return out


class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        use_residual_block=False,
        input_size=None,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then not
                use window attention.
            use_residual_block (bool): If True, use a residual block after the MLP block.
            input_size (int or None): Input resolution for calculating the relative positional
                parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        from timm.models.layers import DropPath, Mlp

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

        self.window_size = window_size

        self.use_residual_block = use_residual_block
        if use_residual_block:
            # Use a residual block with bottleneck channel as dim // 2
            self.residual = ResBottleneckBlock(
                in_channels=dim,
                out_channels=dim,
                bottleneck_channels=dim // 2,
                norm="LN",
                act_layer=act_layer,
            )

    def forward(self, x):
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

        if self.use_residual_block:
            x = self.residual(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x
                                         

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
        act_layer=nn.GELU,
        use_abs_pos=True,
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        window_block_indexes=(),
        residual_block_indexes=(),
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_channel=[128, 256, 768, 768],
        out_ids=11,
        patch_embed="ConvNext",
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

        if patch_embed == "patch_embed":
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

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.window_size = []
        if isinstance(window_size, int):
            for i in range(depth):
                if i in window_block_indexes:
                    self.window_size.append(window_size)
                else:
                    self.window_size.append(0)
        elif len(window_size) > 1:
            assert len(window_size) == len(window_block_indexes)
            index = 0
            for i in range(depth):
                if i in window_block_indexes:
                    self.window_size.append(window_size[index])
                    index += 1
                else:
                    self.window_size.append(0)
        
        else:
            raise NotImplementedError(
                f"window_size size type {type(window_size)} is not supported"
            )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=self.window_size[i],
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            if use_act_checkpoint:
                # TODO: use torch.utils.checkpoint
                from fairscale.nn.checkpoint import checkpoint_wrapper

                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self.out_index = out_index
        self.out_ids = out_ids
        self._out_feature_strides = {}
        self._out_feature_channels = {}
        self._out_features = []

        stride_out = 4
        for i_layer in self.out_index:
            layer = nn.LayerNorm(self.out_channel[i_layer])
            layer_name = f'out_norm{i_layer}'
            self.add_module(layer_name, layer)
            self._out_feature_channels[f"p{i_layer}"] = self.out_channel[i_layer]
            self._out_feature_strides[f"p{i_layer}"] = stride_out
            self._out_features.append(f"p{i_layer}")
            stride_out *= 2
        
        self.add_module("learnable_downsample", learnableCommonConv(embed_dim, embed_dim))

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
            )

            x = x + pos_embed
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        outputs.append(x.permute(0, 3, 1, 2).contiguous())
        outputs.append(self.learnable_downsample(x.permute(0, 3, 1, 2).contiguous()))
        final_results = {}

        for i, out in enumerate(outputs):
            if i in self.out_index:
                out = out.permute(0, 2, 3, 1).contiguous()
                norm_layer = getattr(self, f'out_norm{i}')
                out = norm_layer(out)
                final_results[f'p{i}'] = out.permute(0, 3, 1, 2).contiguous()
        return final_results


class ConvNextWindowViTBase(ViT):
    def __init__(
        self, out_index=[0, 1, 2, 3], out_channel = [128, 256, 768, 768], 
        convnext_pt=False, 
        drop_block=None, 
        window_size=14, 
        window_block_indexes=[3, 4, 6, 7, 9, 10],
        down_sample="common"):
        model_args = dict(
            patch_embed = "ConvNext",
            out_index=out_index, 
            out_channel=out_channel,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_rel_pos=True,
        )
        super(ConvNextWindowViTBase, self).__init__(
           **model_args
        )
        if convnext_pt is True:
            model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
            backbone = _create_hybrid_backbone(pretrained=True, pretrained_strict=False, **model_args)
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=2, embed_dim=768)
        
        if drop_block is not None:
            for i in drop_block:
                self.blocks[i] = nn.Identity()
        

        if down_sample == "DAT":
            self.add_module("learnable_downsample", learnableDAT(
                                                in_dim=768,
                                                out_dim=768,
                                                num_heads=12,
                                                ))
        elif down_sample == "convnext":
            self.add_module("learnable_downsample", learnableConv(
                                                in_dim=768,
                                                out_dim=768,
                                                ))
        elif down_sample == "windowattn":
            self.add_module("learnable_downsample", learnableWindowAttn(
                                                in_dim=768,
                                                out_dim=768,
                                                num_heads=12,
                                                ))
        elif down_sample == "common":
            pass
        else:
            raise NotImplementedError(f"{down_sample} is not supported for learnable_downsample")

class ConvNextWindowViTSmall(ViT):
    def __init__(
        self,
        embed_dim=384, num_heads=6, 
        out_index=[0, 1, 2, 3], out_channel = [128, 256, 384, 384], 
        convnext_pt=False, 
        drop_block=None, 
        window_size=14, 
        window_block_indexes=[3, 4, 6, 7, 9, 10],
        down_sample="common"):
        model_args = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            patch_embed = "ConvNext",
            out_index=out_index, 
            out_channel=out_channel,
            window_size=window_size,
            window_block_indexes=window_block_indexes,
        residual_block_indexes=[],
        use_rel_pos=True,
        )
        super(ConvNextWindowViTSmall, self).__init__(
           **model_args
        )
        if convnext_pt is True:
            model_args = dict(depths=[3, 3], dims=[128, 256, 512, 1024], use_head=False)
            backbone = _create_hybrid_backbone(pretrained=True, pretrained_strict=False, **model_args)
            self.patch_embed = HybridEmbed(backbone=backbone, patch_size=2, embed_dim=384)
        
        if drop_block is not None:
            for i in drop_block:
                self.blocks[i] = nn.Identity()
        

        if down_sample == "DAT":
            self.add_module("learnable_downsample", learnableDAT(
                                                in_dim=384,
                                                out_dim=384,
                                                num_heads=12,
                                                ))
        elif down_sample == "convnext":
            self.add_module("learnable_downsample", learnableConv(
                                                in_dim=384,
                                                out_dim=384,
                                                ))
        elif down_sample == "windowattn":
            self.add_module("learnable_downsample", learnableWindowAttn(
                                                in_dim=384,
                                                out_dim=384,
                                                num_heads=12,
                                                ))
        elif down_sample == "common":
            pass
        else:
            raise NotImplementedError(f"{down_sample} is not supported for learnable_downsample")

if __name__ == "__main__":
    model = ConvNextWindowViT(down_sample="windowattn")
    x = torch.randn(3, 3, 224, 224)
    x = model(x)
    for k, v in x.items():
        print(v.shape)
