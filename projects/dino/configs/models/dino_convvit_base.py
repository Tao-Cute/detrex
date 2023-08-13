from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detrex.modeling.backbone import ConvNextWindowViTBase, ConvNextWindowViTSmall


from .dino_r50 import model

model.backbone = L(ConvNextWindowViTBase)(
    convnext_pt=True, drop_block=None, 
    window_size=14,
    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
    out_index=[1, 2, 3],
    down_sample="common")


model.neck.input_shapes = {
    "p1": ShapeSpec(channels=256),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=768),
}
model.neck.in_features = ["p1", "p2", "p3"]