import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, Softmax, Linear, LayerNorm
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import warnings
from timm.models.layers import DropPath
import torchvision.models as models
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from .builder import build_fuser, FUSERS


@FUSERS.register_module()
class Fusion(nn.Module):
    def __init__(self,
                 bev_channels=256, prior_channels=64, hidden_c=256, align_fusion=True,
                 bev_size=(100,200), img_size=(200,400), patch_size=(10, 10),
                 decoder_layers=3, dropout=0.1,
                 dis=5,
                 num_heads=8, mlp_ratio=4, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm
                 ):
        super(Fusion, self).__init__()
        self.bev_size = bev_size
        self.img_size = img_size
        self.patch_size = patch_size[0]
        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]
        self.bev_channels = bev_channels
        self.prior_channels = prior_channels
        self.hidden_c = hidden_c
        self.drop_out = dropout

        self.patch_embedding = PatchEmbed(bev_in_channels=self.bev_channels, prior_in_channels=self.prior_channels, out_channels=self.hidden_c, img_size=img_size, patch_size=patch_size)
        self.decoder_layers = decoder_layers
        self.decoder = nn.ModuleList([
            DeformableTransformerDecoderLayer(dim=self.hidden_c)
            for i in range(self.decoder_layers)])
        self.expand = nn.Linear(self.hidden_c, (self.patch_size**2)*self.bev_channels, bias=False)
        self.drop = Dropout(self.drop_out)
        self.dis = dis
        self.get_mask = Mask(bev_channels, num_heads, patch_size, self.grid_size, self.dis)
        
        self.align_fusion = align_fusion
        # if self.align_fusion:
        #     self.alignfusion = AlignFusion(self.bev_channels, self.bev_channels//2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.prior2seg = nn.Conv2d(256,256,kernel_size=2, stride=2)
        self.last_conv = nn.Conv2d(512,256,kernel_size=1)

    def forward(self, bev_embed, prior_features):
        # print("bev_embed1:", bev_embed.shape)
        # print("prior_features1:", prior_features.shape)
        bs, _, c = bev_embed.shape
        # print("prior_features2:", prior_features.shape)
        # prior = prior_features.copy()bev_size
        # print("seg shape", segmentation.shape)
        prior_embedding = self.patch_embedding(prior_features)
        # print(bev_embed.shape, prior_embedding.shape)
        query_feat = bev_embed
        reference_points = torch.zeros((self.bev_size[0], self.bev_size[1], 2))
        for x in range(self.bev_size[0]):
            for y in range(self.bev_size[1]):
                reference_points[x][y][0] = x
                reference_points[x][y][1] = y
        reference_points = torch.unsqueeze(reference_points, 0)
        reference_points = reference_points.repeat(bs, 1, 1, 1)
        reference_points = reference_points.view(bs, -1, 2)
        reference_points = torch.unsqueeze(reference_points, 2)
        reference_points = reference_points.to(bev_embed)  # (b, len, 1, 2)

        spatial_shapes = []
        for i in range(1):
            spatial_shape = (self.grid_size[0], self.grid_size[1])
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query_feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        for i in range(self.decoder_layers):
            query_feat = self.decoder[i](query_feat, reference_points, prior_embedding, spatial_shapes, level_start_index)

        # print(x.shape)
        return query_feat
    


class PatchEmbed(nn.Module):
    def __init__(self, bev_in_channels, prior_in_channels, out_channels, img_size=(200,400), patch_size=(10, 10), dropout_rate=0.1):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size[0]

        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.bev_patch_embedding = Conv2d(in_channels=bev_in_channels,
                                          out_channels=out_channels,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.prior_patch_embedding = Conv2d(in_channels=prior_in_channels,
                                            out_channels=out_channels,
                                            kernel_size=self.patch_size,
                                            stride=self.patch_size)

        # To do: 仿照Transfusion，使用PositionEmbeddingLearned，替代可学习参数
        self.position_embedding_bev = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))
        self.position_embedding_prior = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))

        self.dropout = Dropout(dropout_rate)

        # x   [B, C, H, W]
    def forward(self, prior_feature):
        prior_feature = self.prior_patch_embedding(prior_feature)
        prior_feature = prior_feature.flatten(2).transpose(-1, -2)
        prior_embedding = prior_feature + self.position_embedding_prior
        prior_embedding = self.dropout(prior_embedding)

        return prior_embedding


class Mask(nn.Module):
    def __init__(self, c_in, num_heads, patch_size, grid_size, dis):
        super(Mask, self).__init__()
        self.dis = dis
        self.num_heads = num_heads
        self.patch_size = patch_size[0]
        self.grid_size = grid_size

        self.dis_mask = self.get_distance_mask(dis)
        self.conv = Conv2d(in_channels=c_in,
                                          out_channels=1,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
    def forward(self, x):
        dis_mask = self.dis_mask.to(x)
        dis_mask = dis_mask.unsqueeze(0)
        x = self.conv(x)
        # print(x.shape)
        b, _, _, _ = x.shape
        x = x.repeat(1, self.num_heads, 1, 1).flatten(2).view(b*self.num_heads,-1)
        x = x.unsqueeze(1)
        # print(dis_mask.shape)
        # print(x.shape)
        x = x + dis_mask
        # print(x.shape)
        return x
    def get_distance_mask(self, dis):
        mask = torch.zeros((self.grid_size[0], self.grid_size[1], self.grid_size[0], self.grid_size[1]))
        for x1 in range(self.grid_size[0]):
            for y1 in range(self.grid_size[1]):
                for x2 in range(self.grid_size[0]):
                    for y2 in range(self.grid_size[1]):
                        if abs(x1-x2) <= dis and abs(y1-y2) <= dis:
                            mask[x1][y1][x2][y2] = float('-inf')
        return mask.view(self.grid_size[0]*self.grid_size[1], self.grid_size[0]*self.grid_size[1])


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.num_heads = num_heads
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # B,H*W,C

        y = self.norm1(y)

        # print(x.shape, y.shape, attn_mask.shape)
        x = self.cross_attn(x, y, y, attn_mask=attn_mask)[0]

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, dim, n_levels=1, n_points=8, num_heads=8, im2col_step=128,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(DeformableTransformerDecoderLayer, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.cross_attn = MSDeformAttn(dim, n_levels, num_heads, n_points, im2col_step)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        shortcut = tgt
        tgt = self.norm1(tgt)
        src = self.norm1(src)

        tgt = self.cross_attn(tgt, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt = shortcut + self.drop_path(tgt)
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))

        return tgt


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=20, im2col_step=64):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = im2col_step

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        # print(query.shape, reference_points.shape, input_spatial_shapes.shape, input_flatten.shape)
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        # print((input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum())
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # print(input_spatial_shapes[..., 1].shape)
        # print(input_spatial_shapes[..., 0].shape)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MultiScaleDeformableAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0