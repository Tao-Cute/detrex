from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.exp.dino_fan_base_224 import model

# modify training config
train.init_checkpoint = "model_zoo/pretrained_model/fan_vit_base.pth.tar"
train.output_dir = "./output/dino_fanMyConv_base_224_4scale_12ep"
train.checkpointer.max_to_keep=5


