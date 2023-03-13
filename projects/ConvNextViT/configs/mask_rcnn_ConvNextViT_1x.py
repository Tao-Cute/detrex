from .mask_rcnn_r50_3x import model, dataloader, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT, ConvNextWindowViT
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler

model.backbone = L(FPN)(
    bottom_up=L(ConvNextWindowViT)(),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/vit828.ckpt"
train.output_dir = "./output/maskrcnn_convvit8281x"

optimizer.lr = 0.0001
optimizer.weight_decay = 0.05
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
