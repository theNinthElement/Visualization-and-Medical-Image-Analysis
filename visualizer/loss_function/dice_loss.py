import torch.nn as nn
import gin

def get_dice_coefficient(input, target, smooth=1.0):
    input = input.contiguous()
    target = target.contiguous()
    intersection = (input * target).sum(dim=2).sum(dim=2)
    dice_coeff = ((2.0 * intersection + smooth)
            / (input.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
    return dice_coeff

@gin.configurable
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, input, target):
        dice_coeff = get_dice_coefficient(input, target, self.smooth)
        loss = 1 - (dice_coeff)
        return loss.mean()
