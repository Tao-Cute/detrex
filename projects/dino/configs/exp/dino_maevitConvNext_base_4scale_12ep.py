from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from ..models.exp.dino_maevit_ConvNext import model

# modify training config
train.init_checkpoint = "/shiyi_root/ytcheng/ckpts/ConvNext78.442.pth.tar"
train.output_dir = "./output/dino_maevitConvNext78.442_base_4scale_12ep"
