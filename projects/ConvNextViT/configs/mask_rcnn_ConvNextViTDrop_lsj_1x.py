from .mask_rcnn_r50_3x import model, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import ConvNextWindowViT, ViTDrop
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import os
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler

from .common.coco_loader_lsj import dataloader
dataloader.train.total_batch_size = 16

model.backbone = L(FPN)(
    bottom_up=L(ConvNextWindowViT)(drop_block=[0, 1, 2]),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/ViTDrop.ckpt"
root_path = './output/MaskRCNN/'
file_name = root_path + 'EXP' + str(len(os.listdir(root_path)) + 1)
train.output_dir = file_name

optimizer.lr = 0.0002
optimizer.weight_decay = 0.1
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}


train.max_iter = 90000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[60000, 80000, 90000],
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.001,
)
