import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import FUSION_LAYERS
from mmcv.runner.base_module import BaseModule


class ChannelAttention(nn.Module):
    """
    标准的通道注意力模块，用于重新加权通道特征。
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        # 使用自适应池化来聚合空间维度信息
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 使用1x1卷积来学习通道间的关系
        # 注意：这里所有操作共享同一个卷积层权重，这是CBAM中的一个简化设计
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别通过最大池化和平均池化
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        
        # 通过卷积层
        max_conv = self.conv(max_pooled)
        avg_conv = self.conv(avg_pooled)
        
        # 将两个结果相加，并通过sigmoid得到最终的通道注意力权重
        out = max_conv + avg_conv
        attention = self.sigmoid(out)
        
        # 将注意力权重应用到原始特征图上
        return x * attention

class SpatialAttention(nn.Module):
    """
    改进的空间注意力模块。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 【修改】使用更大的卷积核（如7x7）来捕捉空间上下文信息。
        # padding = (kernel_size - 1) // 2 确保特征图尺寸不变。
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度进行最大池化和平均池化，得到两个描述符
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # 将两个描述符拼接起来
        pool = torch.cat([max_pool, avg_pool], dim=1)
        
        # 通过卷积层生成空间注意力图
        conv = self.conv(pool)
        attention = self.sigmoid(conv)
        return attention


@FUSION_LAYERS.register_module()
class UnifiedSpatialAttention(BaseModule):
    """
    修改后的通用特征增强模块。
    """
    def __init__(self, bev_channels=256, prior_channels=256, kernel_size=7):
        super(UnifiedSpatialAttention, self).__init__()
        self.bev_channels = bev_channels
        self.prior_channels = prior_channels
        
        # 输入通道数为两个特征拼接后的总通道数
        total_channels = self.bev_channels + self.prior_channels
        self.channel_attention = ChannelAttention(total_channels)
        
        # 使用改进后的空间注意力模块
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, refined_features, prior_features):
        # 1. 拼接特征
        unified_features = torch.cat([refined_features, prior_features], dim=1)

        # 2. 应用通道注意力，让模型学习哪些通道更重要
        unified_features_ca = self.channel_attention(unified_features)
        
        # 3. 基于通道增强后的特征，生成空间注意力图
        #    这让空间注意力的计算也考虑了通道的重要性
        attention_map = self.spatial_attention(unified_features_ca)
        
        # 4. 【核心修改】使用注意力图作为“门”，自适应地增强两个特征流。
        #    移除 `1 - attention_map` 的强假设抑制逻辑。
        #    模型现在可以学习在重要区域（attention_map接近1）同时增强两个特征，
        #    在不重要区域（attention_map接近0）同时抑制它们。
        # enhanced_refined = refined_features * attention_map
        enhanced_prior = prior_features * attention_map
        
        # 5. 返回经过注意机制增强后的特征拼接结果
        return torch.cat([refined_features, enhanced_prior], dim=1)

