from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.dino_convvit_base import model

# modify training config
train.init_checkpoint = "model_zoo/ConvNextViT_Base.ckpt"
train.output_dir = "./output/dino_ConvNextViT_base_4scale_12ep"