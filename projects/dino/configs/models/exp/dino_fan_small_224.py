from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import fan_small_12_p4_hybrid

from ..dino_r50 import model

model.backbone = L(fan_small_12_p4_hybrid)(
    out_ids=[1, 2 ,3],
    if_vit=True,
    patch_embed="MyConv"
)

model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=384),
    "p3": ShapeSpec(channels=768),
}

model.neck.in_features = ["p1", "p2", "p3"]