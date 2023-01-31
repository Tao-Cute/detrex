from detectron2 import model_zoo
from .models.cascade_vitbase_ConvNext import model
from detectron2.solver import WarmupParamScheduler
from detectron2.config import LazyCall as L
from fvcore.common.param_scheduler import MultiStepParamScheduler

train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "model_zoo/pretrain/R-50.pkl"


from .common.coco_loader import dataloader
dataloader.train.total_batch_size = 16


optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {
    "pos_embed": {"weight_decay": 0.0},
}
optimizer.lr = 1e-4

train.max_iter = 270000
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[210000, 250000, 270000],
    ),
    warmup_length=500 / train.max_iter,
    warmup_factor=0.001,
)