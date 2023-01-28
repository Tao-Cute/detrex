from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.exp.dino_maevit_ConvNext import model

train.init_checkpoint = "vitBase_convnext819.pth"
train.output_dir = "./output/dino_maevitConvNex81.9_base_4scale_12ep"
