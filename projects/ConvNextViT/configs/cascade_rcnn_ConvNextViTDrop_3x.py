from .cascade_rcnn_r50_3x import model, dataloader, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L

from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import ConvNextWindowViT, ViTDrop
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

model.backbone = L(FPN)(
    bottom_up=L(ViTDrop)(drop_block=[0, 1, 2]),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/vit828.ckpt"
train.output_dir = "./output/cascade_convvit828exp1"

optimizer.lr = 0.0001
optimizer.weight_decay = 0.1
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
