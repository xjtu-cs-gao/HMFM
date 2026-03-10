import torch
import torch.nn as nn
import torchvision.models as models
import copy

from mmcv.runner.base_module import BaseModule
from mmdet.models import BACKBONES


class convrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x))


@BACKBONES.register_module()
class ResUNet(BaseModule):
    def __init__(self, outC):
        super(ResUNet, self).__init__()
        
        # 使用本地预训练模型，避免网络下载问题
        self.base_model = models.resnet18(pretrained=False)
        # 加载本地预训练权重
        import os
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), 'ckpts', 'resnet18-f37072fd.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            self.base_model.load_state_dict(state_dict)
        else:
            print(f"Warning: 预训练模型文件不存在: {model_path}")

        self.base_layers = list(self.base_model.children())

        self.layer0 = copy.deepcopy(nn.Sequential(*self.base_layers[:3]))  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = copy.deepcopy(nn.Sequential(*self.base_layers[3:5]))  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = copy.deepcopy(self.base_layers[5])  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 256, 1, 0)
        self.base_layers = None

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, outC, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)

        layer2 = self.layer2_1x1(layer2)          # 256
        x = self.upsample(layer2)                      # 256
        layer1 = self.layer1_1x1(layer1)                    # 64
        x = torch.cat([x, layer1], dim=1)                           # 256 + 64
        x = self.conv_up1(x)                                        # 256

        x = self.upsample(x)                                       # 256 * 100 * 200
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)                                                 # 128 * 200 * 400
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

    def init_weights(self):
        # 仅初始化新加入的解码器与头部卷积层，不覆盖预训练的 ResNet 编码器权重
        modules_to_init = [
            self.layer0_1x1,
            self.layer1_1x1,
            self.layer2_1x1,
            self.conv_up1,
            self.conv_up0,
            self.conv_original_size0,
            self.conv_original_size1,
            self.conv_original_size2,
            self.conv_last,
        ]

        for module_group in modules_to_init:
            for m in module_group.modules() if hasattr(module_group, 'modules') else [module_group]:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)



