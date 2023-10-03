import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}
        return

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, bn_names=None):
        super(BasicBlock, self).__init__()
        self.bn1 = batch_norm_multiple(nn.BatchNorm2d, in_planes, bn_names)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = batch_norm_multiple(nn.BatchNorm2d, out_planes, bn_names)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        bn_name = x[1]
        x = x[0]
        if not self.equalInOut:
            x = self.relu1(self.bn1([x, bn_name]))
        else:
            out = self.relu1(self.bn1([x, bn_name]))
        if self.equalInOut:
            out = self.relu2(self.bn2([self.conv1(out), bn_name]))
        else:
            out = self.relu2(self.bn2([self.conv1(x), bn_name]))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return [torch.add(self.convShortcut(x), out), bn_name]
        else:
            return [torch.add(x, out), bn_name]


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, bn_names=None):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, bn_names)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, bn_names):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, bn_names=bn_names))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, bn_names=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, bn_names)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, bn_names)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, bn_names)
        # global average pooling and classifier
        self.bn1 = batch_norm_multiple(nn.BatchNorm2d, nChannels[3], bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, batch_norm_multiple):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, bn_name):
        out = self.conv1(x)
        out = self.block1([out, bn_name])
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.linear(out, bn_name)

    def get_feature(self, x, bn_name):
        out = self.conv1(x)
        out = self.block1([out, bn_name])
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out