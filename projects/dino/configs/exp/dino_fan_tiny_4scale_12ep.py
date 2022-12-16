from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.exp.dino_fan_tiny_224 import model

# modify training config
train.init_checkpoint = "model_zoo/pretrained_model/fan_hybrid_tiny.pth.tar"
train.output_dir = "./output/dino_fan_tiny_224_4scale_12ep"