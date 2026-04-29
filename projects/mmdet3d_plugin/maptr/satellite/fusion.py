from mmdet3d.models import FUSION_LAYERS
from mmcv.runner.base_module import BaseModule
from mmdet3d.models import builder
import torch.nn.functional as F
import torch.nn as nn
import torch


@FUSION_LAYERS.register_module()
class HMFM(BaseModule):
    def __init__(self, feature_fusion=None, align_fusion=None, bev_fusion=None, seg_backbone=None,
                 bev_channels=256, prior_channels=256, hidden_channels=256,
                 img_size=(100, 200), bev_h=200, bev_w=100,
                ):
        super(HMFM, self).__init__()
        self.bev_channels = bev_channels
        self.prior_channels = prior_channels
        self.hidden_channels = hidden_channels
        self.align_fusion = align_fusion
        self.img_size = img_size
        self.bev_h = bev_h
        self.bev_w = bev_w

        if feature_fusion:
            feature_fusion = feature_fusion.copy()
            feature_fusion.setdefault('bev_channels', bev_channels)
            feature_fusion.setdefault('prior_channels', hidden_channels)
            feature_fusion.setdefault('hidden_c', hidden_channels)
            feature_fusion.setdefault('bev_size', (bev_h, bev_w))
            self.feature_fusion = builder.build_fusion_layer(feature_fusion)
        else:
            self.feature_fusion = None
        if align_fusion:
            align_fusion = align_fusion.copy()
            align_fusion.setdefault('inplane', hidden_channels)
            align_fusion.setdefault('feature2_inplane', bev_channels)
            align_fusion.setdefault('outplane', hidden_channels)
            self.align_fusion = builder.build_fusion_layer(align_fusion)
        else:
            self.align_fusion = None
        if bev_fusion:
            bev_fusion = bev_fusion.copy()
            bev_fusion.setdefault('bev_channels', bev_channels)
            bev_fusion.setdefault('prior_channels', hidden_channels)
            self.bev_fusion = builder.build_fusion_layer(bev_fusion)
        else:
            self.bev_fusion = None
        
        self.conv = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.seg = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(bev_channels + hidden_channels, bev_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(bev_channels),
            nn.ReLU(inplace=True)
        )
        
    # bev_feats: (bs, 200*100, 256), prior_features: (bs, 256, 400, 200)
    def forward(self, bev_feats, prior_features):

        bs, _, c = bev_feats.shape
        bev_features = bev_feats.reshape(bs, self.bev_h, self.bev_w, c).permute(0, 3, 1, 2)  # (bs, 256, 200, 100)

        prior_features = self.conv(prior_features)  # (bs, 256, 200, 100)
        segmentation = self.seg(prior_features)  # (bs, 1, 200, 100)
        if self.feature_fusion:
            refined_features = self.feature_fusion(bev_features, prior_features, segmentation)  # (bs, 256, 200, 100)
        else:
            refined_features = bev_features
        if self.align_fusion:
            prior_features = self.align_fusion(prior_features, refined_features)
        if self.bev_fusion:
            new_bev_features = self.bev_fusion(refined_features, prior_features)  # (bs, 512, 200, 100)
        else:
            new_bev_features = torch.cat([refined_features, prior_features], dim=1)
        new_bev_features = self.conv_last(new_bev_features)  # (bs, 256, 200, 100)
        new_bev_features = new_bev_features.permute(0, 2, 3, 1).reshape(bs, -1, self.bev_channels) # (0, w, h, c) -> (0, w*h, c)
        return new_bev_features
