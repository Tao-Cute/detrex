

from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from .common.coco_loader import dataloader

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"


# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = False
train.ddp.fp16_compression = False
train.init_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"

dataloader.train.total_batch_size = 16

# 36 epochs
train.max_iter = 270000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[210000, 250000, 270000],
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW

optimizer.lr = 0.0001
