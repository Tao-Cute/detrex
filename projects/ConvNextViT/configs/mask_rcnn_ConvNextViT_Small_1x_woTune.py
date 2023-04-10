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
    bottom_up=L(ConvNextWindowViTSmall)(convnext_pt=True, drop_block=[0, 1, 2], 
                                    window_size=14,
                                    window_block_indexes=[3, 4, 6, 7, 9, 10],
                                    down_sample="common"),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/deit_small_patch16_224.pth"
root_path = './output/MaskRCNNSmall/'
file_name = root_path + 'EXP' + str(len(os.listdir(root_path)) + 1)
train.output_dir = file_name

optimizer.lr = 0.0001
optimizer.weight_decay = 0.1
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)


train.max_iter = 90000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[60000, 80000, 90000],
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.001,
)
