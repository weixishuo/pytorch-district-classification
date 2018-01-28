import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

config = {
    'weight_decay': 1e-5,
    'bn_epsilon': 1e-4,
}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_first=True, proj_stride=2):
        super(ResidualBlock, self).__init__()
        # first bn_relu_conv
        if is_first:
            self.conv1 = self._bn_conv_relu('conv1', in_channels, out_channels,
                                            stride=proj_stride, pre_act=False)
        else:
            self.conv1 = self._bn_conv_relu('conv1', in_channels, out_channels)

        self.conv2 = self._bn_conv_relu('conv2', out_channels, out_channels)
        self.short_cut = nn.Sequential()
        if is_first:
            self.short_cut.add_module('proj_conv',
                                      nn.Conv2d(in_channels,out_channels,
                                      kernel_size=1, stride=proj_stride))
            self.short_cut.add_module('proj_bn', nn.BatchNorm2d(out_channels))
            self.short_cut.add_module('proj_relu', nn.ReLU())

    def _bn_conv_relu(self, name, in_channels, out_channels, stride=1, pre_act=True):
        block = nn.Sequential()
        if pre_act:
            block.add_module(name+'_bn', nn.BatchNorm2d(in_channels))
            block.add_module(name+'_relu', nn.ReLU())
        block.add_module(name+'conv', nn.Conv2d(in_channels,
                                                out_channels,
                                                3,
                                                stride=stride,
                                                padding=1))

        return block

    def forward(self, x):
        short_cut = self.short_cut.forward(x)
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)
        return short_cut + x

class Encoder(nn.Module):
    def __init__(self, training=False):
        super(Encoder, self).__init__()
        self.training = training
        self.stage_0 = self.stage('stage_0', 3, 16, 1, 1)
        self.max_pool = nn.MaxPool2d(2)
        self.stage_1 = self.stage('stage_1', 16, 16, 3, 1)
        self.stage_2 = self.stage('stage_2', 16, 32, 4, 2)
        self.stage_3 = self.stage('stage_3', 32, 64, 4, 2)
        self.stage_4 = self.stage('stage_4', 64, 128, 2, 2)
        self.global_avg_pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(128, 2)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.stage_0.forward(x)
        x = self.stage_1.forward(x)
        x = self.stage_2.forward(x)
        x = self.stage_3.forward(x)
        x = self.stage_4.forward(x)
        x = self.global_avg_pool(x)
        x = F.dropout(x, training=self.training)
        x = x.view(-1, 128)
        x = self.linear(x)
        return self.output(x)

    def stage(self, name, in_channels, out_channels, num_layers, proj_stride):
        block = nn.Sequential()
        for l in range(num_layers):
            channels = in_channels if l==0 else out_channels
            block.add_module(name+str(l),
                             ResidualBlock(channels,
                                           out_channels, l==0, proj_stride))

        return block

class Decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass
