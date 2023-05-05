from .cascade_rcnn_r50_3x import model, dataloader, optimizer, lr_multiplier, train
from detectron2.config import LazyCall as L

from detectron2.modeling.backbone import FPN
from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT, ConvNextWindowViTBase, ConvNextWindowViTSmall

from detectron2.modeling.backbone.fpn import LastLevelMaxPool

model.backbone = L(FPN)(
    bottom_up=L(ConvNextWindowViTBase)(convnext_pt=True, drop_block=None, 
                                    window_size=14,
                                    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                                    down_sample="common"),
    in_features=["p0", "p1", "p2", "p3"],
    out_channels=256,
    top_block=L(LastLevelMaxPool)(),
)
train.init_checkpoint = "model_zoo/deit_base_patch16_224.pth"
train.output_dir = "./output/cascade_DeiT_Base_Lr1_Wd0.05"

optimizer.lr = 0.0001
optimizer.weight_decay = 0.05
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}