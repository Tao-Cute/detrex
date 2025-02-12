from .mask_rcnn_r50_3x import model, dataloader, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT, ConvNextWindowViT
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
import os
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.solver import WarmupParamScheduler

model.backbone = L(FPN)(
    bottom_up=L(ConvNextWindowViT)(drop_block=[0, 1, 2], 
                                   window_size=14,
                                   window_block_indexes=[3, 4, 6, 7, 9, 10]),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/ViTDrop.ckpt"
root_path = './output/MaskRCNN/'
file_name = root_path + 'EXP' + str(len(os.listdir(root_path)) + 1)
train.output_dir = file_name

optimizer.lr = 0.0001
optimizer.weight_decay = 0.1
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
