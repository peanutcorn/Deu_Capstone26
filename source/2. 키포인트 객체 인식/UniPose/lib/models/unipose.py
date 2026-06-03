import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
from pprint import pprint

#pprint(sys.path)
#print(os.path.abspath('.'))
#sys.path.insert(1, os.path.abspath('.'))
pprint(sys.path)

from models.modules.wasp import build_wasp
from models.modules.decoder import build_decoder
from models.modules.resnet import build_resnet


class unipose(nn.Module):
    def __init__(self, dataset, backbone='ResNet101', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, stride=8):
        super(unipose, self).__init__()
        self.stride = stride
        self.backbone_name = backbone.lower()
        BatchNorm = nn.BatchNorm2d

        self.num_classes = num_classes

        self.pool_center   = nn.AvgPool2d(kernel_size=9, stride=8, padding=1)

        self.backbone      = build_resnet(name=self.backbone_name, 
                                          BatchNorm=nn.BatchNorm2d, 
                                          pretrained=True, 
                                          output_stride=output_stride)
        self.wasp          = build_wasp(backbone=self.backbone_name, 
                                        output_stride=output_stride, 
                                        BatchNorm=BatchNorm)
        self.decoder       = build_decoder(dataset=dataset, 
                                           num_classes=num_classes, 
                                           backbone=self.backbone_name, 
                                           BatchNorm=BatchNorm)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.wasp(x)
        x = self.decoder(x, low_level_feat)
        if self.stride != 8:
            x = F.interpolate(x, size=(input.size()[2:]), mode='bilinear', align_corners=True)

        # If you are extracting bouding boxes as well
#         return x[:,0:self.num_classes+1,:,:], x[:,self.num_classes+1:,:,:] 
    
        # If you are only extracting keypoints
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":    
    #model = waspnet(backbone='resnet', output_stride=16)
    model = unipose(backbone='resnet101', 
                    dataset='NIA', 
                    output_stride=16,
                    num_classes=17)
    model.eval()
    # input = torch.rand(1, 3, 513, 513)
    input = torch.rand(1, 3, 512, 512)
    output = model(input)
    print(output.size())


