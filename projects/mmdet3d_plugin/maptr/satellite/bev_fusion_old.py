import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import FUSION_LAYERS
from mmcv.runner.base_module import BaseModule



class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        max_conv = self.conv(max_pooled)
        avg_conv = self.conv(avg_pooled)
        out = max_conv + avg_conv
        attention = self.sigmoid(out)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        conv = self.conv(pool)
        attention = self.sigmoid(conv)
        return attention


@FUSION_LAYERS.register_module()
class UnifiedSpatialAttention(BaseModule):
    def __init__(self, bev_channels=256, prior_channels=256, kernel_size=7):
        super(UnifiedSpatialAttention, self).__init__()
        self.bev_channels = bev_channels
        self.prior_channels = prior_channels

        self.channel_attention = ChannelAttention(self.bev_channels+self.prior_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, refined_features, prior_features):
        unified_features = torch.cat([refined_features, prior_features], dim=1)

        unified_features = self.channel_attention(unified_features)
        attention_map = self.spatial_attention(unified_features)
        contrary_attention_map = 1 - attention_map
        
        return torch.cat([refined_features, contrary_attention_map * prior_features], dim=1)