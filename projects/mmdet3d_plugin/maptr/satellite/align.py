import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import FUSION_LAYERS
from mmcv.runner.base_module import BaseModule


def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.Sigmoid()
    )


@FUSION_LAYERS.register_module()
class AlignFusion(BaseModule):
    def __init__(self, inplane=256, outplane=256, kernel_size=3, gate='simple', feature2_inplane=None):
        super(AlignFusion, self).__init__()

        feature2_inplane = feature2_inplane or inplane
        self.down_1 = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_2 = nn.Conv2d(feature2_inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)
        nn.init.constant_(self.flow_make.weight, 0.)
        if gate == 'simple':
            self.gate = SimpleGate(inplane, 2)
        elif gate == "":
            self.gate = None
        else:
            raise ValueError("no this type of gate")
    
    def forward(self, feature1, feature2):
        x1 = feature1
        x2 = feature2
        h, w = x1.size()[2:]
        size = (h, w)

        x1 = self.down_1(x1)
        x2 = self.down_2(x2)

        flow = self.flow_make(torch.cat([x1, x2], 1))
        # print("\n\n\n")
        # print(flow.shape)

        # if self.gate:
        #     flow = self.gate(feature1, flow)
        feature1_warp = self.flow_warp(feature1, flow, size=size)
        return feature1_warp

    
    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = input.new_tensor([[[[max(out_w - 1, 1) / 2.0, max(out_h - 1, 1) / 2.0]]]])
        grid_y = torch.linspace(-1.0, 1.0, out_h, device=input.device, dtype=input.dtype).view(-1, 1).repeat(1, out_w)
        grid_x = torch.linspace(-1.0, 1.0, out_w, device=input.device, dtype=input.dtype).repeat(out_h, 1)
        grid = torch.cat((grid_x.unsqueeze(2), grid_y.unsqueeze(2)), 2)
        grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class SimpleGate(nn.Module):
    def __init__(self, inplane, outplane):
        super(SimpleGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(inplane, 2, 1),
            nn.Sigmoid()
        )


    def forward(self, feats, x):
        size = feats.size()[2:]
        flow_gate = self.gate(feats)
        x = x*flow_gate
        return x
