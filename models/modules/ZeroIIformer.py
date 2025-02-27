import os

import PIL
import cv2
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from matplotlib import pyplot as plt
from thop import profile

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat, einops
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
import numbers
from PIL.Image import Image
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src
def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class TransformerBlock(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers=2, dim=576, num_heads=4, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', withZero=0):
        # 2048
        super().__init__()

        self.withZero = withZero
        if self.withZero == 0:
            self.layer_stack = nn.ModuleList([
                TransformerBlock_withZeromap(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(n_layers)])
        else:
            self.layer_stack = nn.ModuleList([
                TransformerBlock_noZeromap(dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
                for _ in range(n_layers)])

    def forward(self, x, mask):
        for enc_layer in self.layer_stack:
            x = enc_layer(x, mask)
        return x

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class TransformerBlock_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_withZeromap, self).__init__()

        self.attn_intra = Attention_withZeromap(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        # m = self.attn_inter(x, zero_map)
        # z = x + self.attn_intra(m, zero_map)
        z = self.attn_intra(x, zero_map)

        out = z + self.ffn(self.norm2(z))

        return out

class TransformerBlock_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias'):
        super(TransformerBlock_noZeromap, self).__init__()

        self.attn_intra = Attention_noZeromap(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, zero_map):
        # m = self.attn_inter(x, None)
        # z = x + self.attn_intra(m)
        z = self.attn_intra(x)

        out = z + self.ffn(self.norm2(z))

        return out

class Attention_withZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias, N=8):
        super(Attention_withZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1))
        self.temperature3 = nn.Parameter(torch.ones(1, 1))
        self.temperature4 = nn.Parameter(torch.ones(1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, zero_map):
        #采用3之W×W、max方案
        zero_map[zero_map <= 0.2] = 0
        zero_map[zero_map > 0.2] = 1
        zero_map = F.interpolate(zero_map, (x.shape[2], x.shape[3]), mode='bilinear')
        x_copy = x
        x = self.norm1(x)
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q_1, k_1, v_1 = qkv.chunk(3, dim=1)#分出feature的QKV

        q_1 = torch.mean(q_1, 2).unsqueeze(1)
        k_1 = torch.max(k_1, 2)[0].unsqueeze(1)

        q_z1 = torch.mean(zero_map, 2).unsqueeze(1)
        k_z1 = torch.mean(zero_map, 2).unsqueeze(1)

        q_1 = torch.nn.functional.normalize(q_1, dim=-1)
        k_1 = torch.nn.functional.normalize(k_1, dim=-1)

        attn_1 = (q_1.transpose(-1, -2) @ k_1) * self.temperature1 #Q变为W×C，乘以K之C×W(W×C×C×W)得W×W
        attn_z1 = (q_z1.transpose(-1, -2) @ k_z1) * self.temperature2 #Q变为W×C，乘以K之C×W(W×C×C×W)得W×W

        attn_1 = attn_1.softmax(dim=-1)
        attn_z1 = attn_z1.softmax(dim=-1)
        attn_1 = attn_1 + attn_z1
        out_1 = v_1 @ attn_1

        out_1 = self.project_out(out_1)

        out_1 = x_copy + out_1

        #采用1之H×H方案
        out_copy = out_1
        out_1 = self.norm1(out_1)
        out = self.qkv(out_1)
        qkv = self.qkv_dwconv(out)
        q, k, v = qkv.chunk(3, dim=1)
        q = torch.mean(q, 1).unsqueeze(1)
        k = torch.mean(k, 1).unsqueeze(1)

        q_z = torch.mean(zero_map, 1).unsqueeze(1)
        k_z = torch.mean(zero_map, 1).unsqueeze(1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature3
        attn = attn.softmax(dim=-1)
        attn_z = (q_z @ k_z.transpose(-1, -2)) * self.temperature4
        attn_z = attn_z.softmax(dim=-1)
        attn = attn + attn_z
        out = v.transpose(-1, -2) @ attn
        out = out.transpose(-1, -2)

        out = self.project_out(out)

        out = out + out_copy

        return out

class Attention_noZeromap(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_noZeromap, self).__init__()

        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(1, 1))
        self.temperature2 = nn.Parameter(torch.ones(1, 1))

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, ):
        #采用3之W×W、max方案
        x_copy = x
        x = self.norm1(x)
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q_1, k_1, v_1 = qkv.chunk(3, dim=1)#分出QKV

        q_1 = torch.mean(q_1, 2).unsqueeze(1)
        k_1 = torch.max(k_1, 2)[0].unsqueeze(1)

        q_1 = torch.nn.functional.normalize(q_1, dim=-1)
        k_1 = torch.nn.functional.normalize(k_1, dim=-1)

        attn_1 = (q_1.transpose(-1, -2) @ k_1) * self.temperature1 #Q变为W×C，乘以K之C×W(W×C×C×W)得W×W

        attn_1 = attn_1.softmax(dim=-1)
        out_1 = v_1 @ attn_1
        out_1 = self.project_out(out_1)
        out_1 = x_copy + out_1

        #采用1之H×H方案
        out_copy = out_1
        out_1 = self.norm1(out_1)
        out = self.qkv(out_1)
        qkv = self.qkv_dwconv(out)
        q, k, v = qkv.chunk(3, dim=1)
        q = torch.mean(q, 1).unsqueeze(1)
        k = torch.mean(k, 1).unsqueeze(1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-1, -2)) * self.temperature2
        attn = attn.softmax(dim=-1)
        out = v.transpose(-1, -2) @ attn
        out = out.transpose(-1, -2)

        out = self.project_out(out)
        out = out + out_copy

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=dim * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv2d(dim, int(dim / 4), 1, 1, 0, bias=bias)
        self.conv3 = nn.Conv2d(dim, int(dim / 4), 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(dim, int(dim / 4), 5, 1, 2, bias=bias)
        self.conv7 = nn.Conv2d(dim, int(dim / 4), 7, 1, 3, bias=bias)
        #self.conv9 = nn.Conv2d(dim, int(dim / 4), 9, 1, 4, bias=bias)
        self.conv_final = nn.Conv2d(dim, dim, 1, 1, 0, bias=bias, groups = 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x2_1 = self.conv1(x2)
        x2_2 = self.conv3(x2)
        x2_3 = self.conv5(x2)
        x2_4 = self.conv7(x2)
        x2 = torch.cat((x2_1, x2_2, x2_3, x2_4), 1)
        x2 = self.conv_final(x2)
        x2 = self.sigmoid(x2)

        x = F.gelu(x1) * x2
        # x = self.project_out(x)
        return x

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x)
        return out


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)#.flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class former(nn.Module):
    def __init__(self, dd_in=3, embed_dim=48, drop_rate=0.,
                 num_blocks=[2, 2, 2, 2],  # [4, 6, 6, 8]
                 heads=[2, 2, 4, 8],
                 num_refinement_blocks=2,  # 4
                 ffn_expansion_factor=2.6667,
                 bias=False, LayerNorm_type='WithBias'
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)

        self.encoder_level1 = TransformerBlock(dim=embed_dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                                 bias=bias, LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0])

        self.down1_2 = Downsample(embed_dim, embed_dim*2)  ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[1],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1])

        self.down2_3 = Downsample(int(embed_dim * 2 ** 1), int(embed_dim * 2 ** 1*2))  ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(dim=int(embed_dim * 2 ** 2), num_heads=heads[2],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[2])

        self.down3_4 = Downsample(int(embed_dim * 2 ** 2), int(2*embed_dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = TransformerBlock(dim=int(embed_dim * 2 ** 3), num_heads=heads[3],
                                         ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                         LayerNorm_type=LayerNorm_type, n_layers=num_blocks[3], withZero=1)

        self.up4_3 = Upsample(int(embed_dim * 2 ** 3), int(embed_dim * 2 ** 3)//2)  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(embed_dim * 2 ** 3), int(embed_dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = TransformerBlock(dim=int(embed_dim * 2 ** 2), num_heads=heads[2],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[2], withZero=1)

        self.up3_2 = Upsample(int(embed_dim * 2 ** 2), int(embed_dim * 2 ** 2)//2)  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(embed_dim * 2 ** 2), int(embed_dim * 2 ** 1), kernel_size=1, bias=True)
        self.decoder_level2 = TransformerBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[1],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[1], withZero=1)

        self.up2_1 = Upsample(int(embed_dim * 2 ** 1), int(embed_dim * 2 ** 1)//2)

        self.decoder_level1 = TransformerBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[0],
                                                 ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                 LayerNorm_type=LayerNorm_type, n_layers=num_blocks[0], withZero=1)

        self.refinement = TransformerBlock(dim=int(embed_dim * 2 ** 1), num_heads=heads[0],
                                             ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type, n_layers=num_refinement_blocks, withZero=1)

        self.reduce_chan_level00 = nn.Conv2d(int(embed_dim * 8), int(embed_dim * 4), kernel_size=1, bias=True)
        self.reduce_chan_level11 = nn.Conv2d(int(embed_dim * 4), int(embed_dim * 4), kernel_size=1, bias=True)
        self.reduce_chan_level22 = nn.Conv2d(int(embed_dim * 2), int(embed_dim * 4), kernel_size=1, bias=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask=None):
        result = {}
        b, _, _, _ = x.shape
        zero_map = mask.expand(b, self.embed_dim, -1, -1)
        y = self.input_proj(x)
        y = self.pos_drop(y)

        out_enc_level1 = self.encoder_level1(y, zero_map)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        zero_1 = downmask(mask, int(2))  # 相应下采样
        # zero_1 = self.down1_2(zero_map)

        out_enc_level2 = self.encoder_level2(inp_enc_level2, zero_1)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        zero_2 = downmask(mask, int(2 ** 2))
        # zero_2 = self.down2_3(zero_1)

        out_enc_level3 = self.encoder_level3(inp_enc_level3, zero_2)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        zero_3 = downmask(mask, int(2 ** 3))
        latent = self.latent(inp_enc_level4, zero_3)

        result0 = self.reduce_chan_level00(latent)
        result['fea_up0'] = result0

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3, None)
        result1 = self.reduce_chan_level11(out_dec_level3)
        result['fea_up1'] = result1

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2, None)
        result2 = self.reduce_chan_level22(out_dec_level2)
        result['fea_up2'] = result2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1, None)

        out_dec_level1 = self.refinement(out_dec_level1, None)
        result['cat_f'] = out_dec_level1
        return result

def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        assert img.dtype == np.uint8
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return (to_4d(to_CHW(img))) / 127.5 - 1

def downmask(mask, k):
    _, _, h, w = mask.shape

    m = torch.squeeze(mask, 1)
    b, h, w = m.shape
    m = torch.chunk(mask, b, dim=0)[0]
    m = torch.squeeze(m)
    size = (int(h // k), int(w // k))
    mc = m.cpu()
    m_n = mc.numpy()
    m = cv2.resize(m_n, size, interpolation=cv2.INTER_LINEAR)
    m[m <= 0.2] = 0
    m[m > 0.2] = 1
    m = torch.from_numpy(m)

    m = torch.unsqueeze(m, 0)
    m = m.expand(b, 48 * k, -1, -1)
    out_mask = m.cuda()
    return out_mask

if __name__ == "__main__":
    inputs = torch.randn([1, 3, 600, 400])
    net = former(dd_in=3,embed_dim=48)  # 定义好的网络模型
    inputs = torch.randn(1, 3, 512, 512)
    inputs1 = torch.randn(1, 1, 512, 512)
    device = torch.device('cuda:0')
    inputs = inputs.to(device)
    inputs1 = inputs1.to(device)
    flops, params = profile(net.cuda(), (inputs,inputs1,))
    print('flops: ', flops, 'params: ', params)

# if __name__ == '__main__':
#     x = torch.randn([1, 48, 400, 600])
#     zero = torch.randn([1, 48, 400, 600])
#     #torch.max(x, dim=1, keepdim=True)
#     # x = torch.mean(x, 1).unsqueeze(1)
#     mode = Attention_withZeromap(dim=48,  num_heads=4, bias=True)
#     y = mode(x,zero)
#     print(y.shape)

