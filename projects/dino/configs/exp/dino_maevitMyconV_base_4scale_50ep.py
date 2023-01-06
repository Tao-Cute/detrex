from detrex.config import get_config
from ..dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
)
from ..models.exp.dino_maevit_MyConv import model
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_50ep

# modify training config
train.max_iter = 375000
train.init_checkpoint = "model_zoo/pretrained_model/mae_pretrain_vit_base.pth"
train.output_dir = "./output/dino_maevitMyconv_base_4scale_50ep"
