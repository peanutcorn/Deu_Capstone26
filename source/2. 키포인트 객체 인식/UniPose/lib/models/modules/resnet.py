# ------------------------------------------------------------------------------
# Copyright (c) Southeast University. Licensed under the MIT License.
# Written by Sen Yang (yangsenius@seu.edu.cn)
# Modifier : MSP (parkms@markany.com)
# ------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import logging

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, name='resnet101'):
        self.inplanes = 64
        self.name = name
        if self.name not in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            print("=====> not supported %s" % (self.name) )
            logger.error("=====> not supported %s" % (self.name) )
            raise NotImplementedError
            
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()
            
        logger.info("=====> success to load %s" % (self.name) )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        if self.name == 'resnet18':
            #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
            pretrain_dict = torch.load('./weight/resnet18-5c106cde.pth')
        elif self.name == 'resnet34':
            #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-333f7ec4.pth')
            pretrain_dict = torch.load('./weight/resnet34-333f7ec4.pth')
        elif self.name == 'resnet50':
            #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
            pretrain_dict = torch.load('./weight/resnet50-19c8e357.pth')
        elif self.name == 'resnet101':
            #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            pretrain_dict = torch.load('./weight/resnet101-5d3b4d8f.pth')
        elif self.name == 'resnet152':
            #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth')
            pretrain_dict = torch.load('./weight/resnet152-b121ed2d.pth')
        
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

def build_resnet(output_stride, BatchNorm, name='resnet101', pretrained=True):
    if name == 'resnet18':
        model = ResNet(BasicBlock, [2, 2, 2, 2], output_stride, BatchNorm, name=name, pretrained=pretrained)
    elif name == 'resnet34':
        model = ResNet(BasicBlock, [3, 4, 6, 3], output_stride, BatchNorm, name=name, pretrained=pretrained)
    elif name == 'resnet50':
        model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, name=name, pretrained=pretrained)
    elif name == 'resnet101':
        model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, name=name, pretrained=pretrained)
    elif name == 'resnet152':
        model = ResNet(Bottleneck, [3, 8, 36, 3], output_stride, BatchNorm, name=name, pretrained=pretrained)
    else:
        print("=====> not supported %s" % (name) )
        logger.error("=====> not supported %s" % (name) )
        raise NotImplementedError        
    
    return model



if __name__ == "__main__":
    import torch
    model = build_resnet(output_stride=8, BatchNorm=nn.BatchNorm2d, name='ResNet152', pretrained=True)

    input = torch.rand(1, 3, 512, 512)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
