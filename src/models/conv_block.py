from torch import nn
from torch.nn import BatchNorm2d, GELU


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, dropout=0):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = BatchNorm2d(out_channels)
        self.gelu = GELU()

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
        self.bn2 = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1
        x = self.gelu(x)
        x = self.dropout(x)
        return x
