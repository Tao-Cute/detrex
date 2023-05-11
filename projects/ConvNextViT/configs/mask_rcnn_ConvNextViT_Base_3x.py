from .mask_rcnn_r50_3x import model, dataloader, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT, ConvNextWindowViTBase, ConvNextWindowViTSmall
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import os
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler
from functools import partial
from .common.lr_decay import get_vit_lr_decay_rate

model.backbone = L(FPN)(
    bottom_up=L(ConvNextWindowViTBase)(convnext_pt=False, drop_block=None, 
                                    window_size=14,
                                    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                                    down_sample="common"),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/ConvNextViT_Base.ckpt"
file_name = "./output/ConvNextViT_Base_Tune_lr15e-4_3x"
train.output_dir = file_name

optimizer.lr = 0.00015
optimizer.weight_decay = 0.1
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
