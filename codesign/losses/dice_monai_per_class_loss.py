import torch
from .dice_monai_loss import DiceMonaiLoss
from monai.metrics import DiceMetric
from codesign.utils.seg_argmax_pred import seg_argmax_pred

class DiceMonaiPerClass(DiceMonaiLoss):
    def __init__(self, selected_class=0):
        super().__init__()
        self.dice_monai_metric = DiceMetric(include_background=False, reduction="mean")
        self.selected_class = selected_class

    def forward(self, pred, gt):
        binary_pred = seg_argmax_pred(pred, chan_dim=1)
        # get selected class
        class_pred = binary_pred[:, [self.selected_class+1], :, :]
        class_gt = gt[:, [self.selected_class+1], :, :].type(torch.int64)
        # add background (the opposite of the selected-class prediction)
        class_pred = torch.cat([torch.ones_like(class_pred)-class_pred, class_pred], dim=1)
        class_gt = torch.cat([torch.ones_like(class_gt)-class_gt, class_gt], dim=1)
        self.dice_monai_metric(class_pred, class_gt)
        score = self.dice_monai_metric.aggregate()
        self.dice_monai_metric.reset()
        return score

    @property
    def name(self):
        return f"dice_monai_score_class={self.selected_class}"
    
    @property
    def mode(self):
        return "max"