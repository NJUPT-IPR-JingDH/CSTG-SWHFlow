import math
import torch
from torch import nn as nn
import torch.nn.functional as F
from models.modules import thops
from models.modules.flow import Conv2d, Conv2dZeros
from utils.util import opt_get
from functools import reduce
from models.modules.ZeroIIformer import LayerNorm

######特别重要
class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt,fFeatures_firstConv):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)####需要修改
        self.kernel_hidden = 1
        self.affine_eps = 0.0003
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels
        self.hidden_channels_min = 32
        self.fFeatures_firstConv=fFeatures_firstConv
        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.0003)
        self.opt = opt
        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn
        self.MFL = MFL(dim = fFeatures_firstConv)

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine_1 = SWHF(in_channels=self.channels_for_nn + fFeatures_firstConv,#6+288=294->12
                              out_channels=self.channels_for_co * 2,#6* 2
                              hidden_channels=self.hidden_channels,#64
                              kernel_hidden=self.kernel_hidden,#1
                              n_hidden_layers=self.n_hidden_layers)#1
        self.fFeatures_1 = SWHF(in_channels=fFeatures_firstConv,#288->24
                                out_channels=self.in_channels * 2,#12*2
                                hidden_channels=self.hidden_channels,#64
                                kernel_hidden=self.kernel_hidden,#1
                                n_hidden_layers=self.n_hidden_layers)#1

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None):
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)
            # 333
            # Affine Injector
            scaleFt_1, shiftFt_1 = self.feature_extract(self.MFL(ft), self.fFeatures_1)
            z = z + shiftFt_1
            z = z * scaleFt_1
            logdet = logdet + self.get_logdet(scaleFt_1)
            # 444
            # Conditional Affine Coupling
            z1, z2 = self.split(z)
            scale_1, shift_1 = self.feature_extract_aff(z1, self.MFL(ft), self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z2 = z2 + shift_1
            z2 = z2 * scale_1
            logdet = logdet + self.get_logdet(scale_1)
            z = thops.cat_feature(z1, z2, "norm")

            output = z
        else:
            z = input
            # 444
            # Conditional Affine Coupling
            z1, z2 = self.split(z)
            scale_1, shift_1 = self.feature_extract_aff(z1, self.MFL(ft), self.fAffine_1)
            self.asserts(scale_1, shift_1, z1, z2)
            z2 = z2 / scale_1
            z2 = z2 - shift_1
            z = thops.cat_feature(z1, z2, "norm")
            logdet = logdet - self.get_logdet(scale_1)
            # 333
            # Feature Conditional
            scaleFt_1, shiftFt_1 = self.feature_extract(self.MFL(ft), self.fFeatures_1)
            z = z / scaleFt_1
            z = z - shiftFt_1
            logdet = logdet - self.get_logdet(scaleFt_1)

            output = z

        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    ######有用
    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    ######有用
    def feature_extract(self, z, f):
        #########重要
        h = f(z)  # z=8,320,96,96, h=8,24,96,96
        # h = h + z
        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_one(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        output = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return output

    ######有用
    def feature_extract_aff(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def feature_extract_aff_one(self, z1, ft, f):
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        output = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return output

    def feature_extract_single_aff(self, z1, f):
        h = f(z1)

        shift, scale = thops.split_feature(h, "cross")
        scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
        return scale, shift

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)

###############################################################################################################
class SWHF(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, image_shape=None, H=None, W=None):
        super(SWHF, self).__init__()
        layers_1 = [Conv2d(in_channels, hidden_channels, kernel_size=kernel_hidden), nn.GELU()]
        for _ in range(n_hidden_layers):
            layers_1.append(Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_hidden))
            layers_1.append(nn.GELU())
        self.model_1 = nn.Sequential(*layers_1)
        self.SFA = ChannelGate_C(gate_channels=1)
        self.WFA = ChannelGate_WH(gate_channels=hidden_channels, pool_types=('w_max', 'w_avg'))
        self.HFA = ChannelGate_WH(gate_channels=hidden_channels, pool_types=('h_max', 'h_avg'))
        layers_2 = []
        layers_2.append(self.SFA)
        layers_2.append(self.WFA)
        layers_2.append(self.HFA)
        self.model_2 = nn.Sequential(*layers_2)
        self.convzeros = Conv2dZeros(hidden_channels, out_channels, kernel_size=kernel_hidden)

    def forward(self,x):
        x = self.model_1(x)
        x = x + self.model_2(x)
        x = self.convzeros(x)
        return x


class MFL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.6667, bias=True):
        super(MFL, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d(dim, int(dim / 4), 1, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(dim, int(dim / 4), 3, 1, 1, groups=int(dim / 4), bias=bias)
        self.conv5 = nn.Conv2d(dim, int(dim / 4), 5, 1, 2, groups=int(dim / 4), bias=bias)
        self.conv7 = nn.Conv2d(dim, int(dim / 4), 7, 1, 3, groups=int(dim / 4), bias=bias)
        self.conv9 = nn.Conv2d(dim, int(dim / 4), 9, 1, 4, groups=int(dim / 4), bias=bias)

        self.conv_final = nn.Conv2d(2, 1, 5, 1, 2, bias=bias)
        self.norm = LayerNorm(dim, 'WithBias')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_original = x
        x1 = self.conv3(x)
        x2 = self.conv5(x)
        x3 = self.conv7(x)
        x4 = self.conv9(x)
        # x = torch.cat((x1, x2, x3, x4), dim=1)
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        x_cat = self.norm(x_cat)
        x_cat = torch.cat((torch.max(x_cat, 1)[0].unsqueeze(1), torch.mean(x_cat, 1).unsqueeze(1)) , dim=1)
        x = self.conv_final(x_cat)
        x = self.sigmoid(x)
        x = x * x_original

        return x


class ChannelGate_C(nn.Module):
    def __init__(self, gate_channels):
        super(ChannelGate_C, self).__init__()
        self.gate_channels = gate_channels
        self.conv = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels * 2, kernel_size=3, stride=1, padding=1,
                                    groups=gate_channels * 2, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1, bias=True)
        )

    def forward(self, x):
        scale_c_max = torch.max(x, 1)[0].unsqueeze(1)
        scale_c_max = self.conv(scale_c_max)
        scale_c_mean = torch.mean(x, 1).unsqueeze(1)
        scale_c_mean = self.conv(scale_c_mean)
        scale = torch.sigmoid(scale_c_max + scale_c_mean)
        return x * scale


class ChannelGate_WH(nn.Module):##修正版
    def __init__(self, gate_channels, pool_types):
        super(ChannelGate_WH, self).__init__()
        self.gate_channels = gate_channels
        self.conv = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels * 2, kernel_size=3, stride=1, padding=1,
                                    groups=gate_channels * 2, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1, bias=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        for pool_type in self.pool_types:
            if pool_type == 'h_avg':
                h_avg_pool = F.avg_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                h_avg_pool = self.conv(h_avg_pool)
                scale_avg = h_avg_pool
            elif pool_type == 'w_max':
                w_max_pool = F.max_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                w_max_pool = self.conv(w_max_pool)
                scale_max = w_max_pool
            elif pool_type == 'h_max':
                h_max_pool = F.max_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                h_max_pool = self.conv(h_max_pool)
                scale_max = h_max_pool
            elif pool_type == 'w_avg':
                w_avg_pool = F.avg_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                w_avg_pool = self.conv(w_avg_pool)
                scale_avg = w_avg_pool

        scale = torch.sigmoid(scale_avg + scale_max).expand_as(x)

        return x * scale
###############################################################################################################
class NN_F(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1, image_shape=None, H=None, W=None):
        super(NN_F, self).__init__()
        layers = [Conv2d(in_channels, hidden_channels, kernel_size=kernel_hidden), nn.GELU()]
        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_hidden))
            layers.append(nn.GELU())
        layers.append(HWAC(gate_channels=hidden_channels))
        layers.append(Conv2dZeros(hidden_channels, out_channels, kernel_size=kernel_hidden))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        # self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        # if self.bn is not None:
        #     x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate_conv(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=('avg', 'max')):
        super(ChannelGate_conv, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        channel_att_raw = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale
class ChannelGate_conv_wh(nn.Module):
    def __init__(self, gate_channels, pool_types=('avg', 'max')):
        super(ChannelGate_conv_wh, self).__init__()
        self.gate_channels = gate_channels
        self.conv = nn.Sequential(
            nn.Conv2d(gate_channels, gate_channels * 2, kernel_size=1, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels * 2, kernel_size=3, stride=1, padding=1,
                                    groups=gate_channels * 2, bias=True),
            nn.Conv2d(gate_channels * 2, gate_channels, kernel_size=1, bias=True)
        )
        self.pool_types = pool_types

    def forward(self, x):
        for pool_type in self.pool_types:
            if pool_type == 'w_avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                channel_att_raw = self.conv(avg_pool)
                scale_w = torch.sigmoid(channel_att_raw).expand_as(x)
            elif pool_type == 'h_max':
                avg_pool11 = F.max_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                channel_att_raw1 = self.conv(avg_pool11)
                scale_h = torch.sigmoid(channel_att_raw1).expand_as(x)
            elif pool_type == 'w_max':
                avg_pool = F.max_pool2d(x, (x.size(2), 1), stride=(x.size(2), 1))
                channel_att_raw = self.conv(avg_pool)
                scale_w = torch.sigmoid(channel_att_raw).expand_as(x)
            elif pool_type == 'h_avg':
                avg_pool11 = F.avg_pool2d(x, (1, x.size(3)), stride=(1, x.size(3)))
                channel_att_raw1 = self.conv(avg_pool11)
                scale_h = torch.sigmoid(channel_att_raw1).expand_as(x)
        scale = scale_w+scale_h

        return x * scale
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class HWAC(nn.Module):
    def __init__(self, gate_channels):
        super(HWAC, self).__init__()
        self.ChannelGate_wh = ChannelGate_conv_wh(gate_channels, pool_types=('w_avg', 'h_max'))
        self.ChannelGate_wh1 = ChannelGate_conv_wh(gate_channels, pool_types=('w_max', 'h_avg'))

    def forward(self, x):
        x_out_1 = self.ChannelGate_wh(x)
        x_out_1 = self.ChannelGate_wh1(x_out_1)

        return x_out_1+x

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=('avg', 'max'), no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate_conv(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out + x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()
        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


class ECA_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out


# class SE_block(nn.Module):
#     def __init__(self, channel, scaling=4):  # scaling为缩放比例，
#         # 用来控制两个全连接层中间神经网络神经元的个数，一般设置为16，具体可以根据需要微调
#         super(SE_block, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // scaling, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // scaling, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

if __name__ == '__main__':
    mode = ChannelGate_C(1)
    x = torch.randn([4, 64, 64, 64])
    y = mode(x)
    # print(mode)
    # print(y)
    # print("Parameters of full network %.4f " % (sum([m.numel() for m in mode.parameters()]) / 1e6))#25`11