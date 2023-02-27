from detectron2.config import LazyCall as L
from projects.cascadeRCNN.modeling import FPN
from detectron2.layers import ShapeSpec
from detectron2 import model_zoo
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT
from .cascade_r50 import model

model.backbone = L(FPN)(
    bottom_up=L(ConvNextViT)(),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
