from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.exp.dino_fan_large_224 import model

# modify training config
train.init_checkpoint = "model_zoo/pretrained_model/mae_pretrain_vit_base.pth"
train.output_dir = "./output/dino_fan_large_224_4scale_12ep"