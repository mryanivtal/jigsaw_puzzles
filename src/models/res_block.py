from torch import nn


class ResBlock(nn.Module):
    def __init__(self, block, residual_weight=0.5):
        self.residual_weight = residual_weight
        super(ResBlock, self).__init__()

        self.block = block

    def forward(self, x):
        block_result = self.block(x)
        result = self.residual_weight * x + (1 - self.residual_weight) * block_result
        return result
