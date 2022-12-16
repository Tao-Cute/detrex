from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import MyConvViT, MIMConvViT

from ..dino_r50 import model

model.backbone = L(MyConvViT)(
    out_index = [1, 2, 3],
)

model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=768),
}
model.neck.in_features = ["p1", "p2", "p3"]



