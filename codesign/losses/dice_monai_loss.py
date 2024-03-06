from torch import nn 
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from codesign.utils.seg_argmax_pred import seg_argmax_pred

class DiceMonaiLoss(nn.Module):
    def __init__(self, include_background=False):
        super().__init__()
        self.dice_monai_loss = DiceLoss(include_background=include_background, softmax=True)
    
    @property
    def name(self):
        return "dice_monai_loss"

    @property
    def mode(self):
        return "min"
    
    @property
    def task(self):
        return "seg"

    def forward(self, pred, gt):
        return self.dice_monai_loss(pred, gt)

class DiceCEMonaiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_monai_loss = DiceCELoss(include_background=False, softmax=True)
    
    @property
    def name(self):
        return "dice_ce_monai_loss"

    @property
    def mode(self):
        return "min"
    
    @property
    def task(self):
        return "seg"

    def forward(self, pred, gt):
        return self.dice_monai_loss(pred, gt.float())

class DiceMonai(DiceMonaiLoss):
    def __init__(self, include_background=False, ignore_empty=False):
        super().__init__()
        self.dice_monai_metric = DiceMetric(
            include_background=include_background, 
            ignore_empty=ignore_empty,
            reduction="mean",
        )

    def forward(self, pred, gt):
        binary_pred = seg_argmax_pred(pred, chan_dim=1)
        self.dice_monai_metric(binary_pred, gt)
        score = self.dice_monai_metric.aggregate()
        self.dice_monai_metric.reset()
        return score

    @property
    def name(self):
        return "dice_monai_score"
    
    @property
    def mode(self):
        return "max"

class FlattenedDiceMonaiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_monai_loss = DiceLoss(sigmoid=True)
    
    @property
    def name(self):
        return "flattened_dice_monai_loss"

    @property
    def mode(self):
        return "min"
    
    @property
    def task(self):
        return "seg"

    def forward(self, pred, gt):
        pred = pred.reshape(pred.shape[0], 1, -1)
        gt = gt.reshape(gt.shape[0], 1, -1)
        return self.dice_monai_loss(pred, gt)

class FlattenedDiceMonai(FlattenedDiceMonaiLoss):
    def forward(self, pred, gt):
        return 1 - super().forward(pred, gt)

    @property
    def name(self):
        return "flattened_dice_monai_score"
    
    @property
    def mode(self):
        return "max"