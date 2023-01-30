# from detectron2.config import LazyCall as L
# from projects.cascadeRCNN.modeling import FPN
# from detectron2.layers import ShapeSpec
# from detectron2 import model_zoo
# from detectron2.modeling.backbone.fpn import LastLevelMaxPool
# from detrex.modeling.backbone import MyConvViT, MIMConvViT, ConvNextViT

# model = model_zoo.get_config("common/models/cascade_rcnn.py").model
# constants = model_zoo.get_config("common/data/constants.py").constants
# model.pixel_mean = constants.imagenet_rgb256_mean
# model.pixel_std = constants.imagenet_rgb256_std

# model.backbone = L(FPN)(
#     bottom_up=L(ConvNextViT),
#     in_features=["p0", "p1", "p2", "p3"],
#     out_channels=256,
#     top_block=L(LastLevelMaxPool)(),
# )

# from IPython import embed; embed()