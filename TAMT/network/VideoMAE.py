# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import time

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        **kwargs
    }  
class PSRP(nn.Module):
    def __init__(self, channel=768, reduction=16):
        super(PSRP, self).__init__()
        
        self.down = nn.Linear(channel, reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up_weight = nn.Linear(reduction, channel, bias=False)
        self.up_bias = nn.Linear(reduction, channel, bias=False)

    def forward(self, x):
        weight = self.up_weight(self.relu(self.down(x)))
        bias = self.up_bias(self.relu(self.down(x)))

        return weight, bias
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
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
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x
class CosAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # self.scale = qk_scale or head_dim**-0.5
        # DO NOT RENAME [self.scale] (for no weight decay)
        if qk_scale is None:
            self.scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))),
                requires_grad=True)
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (
            F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))

        # torch.log(torch.tensor(1. / 0.01)) = 4.6052
        logit_scale = torch.clamp(self.scale, max=4.6052).exp()

        attn = attn * logit_scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print(x.shape)#torch.Size([85, 1568, 384])
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp_Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return xs + x

class TAA(nn.Module):
    def __init__(self,i,
                 in_channels,
                 n_segment,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAA, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.i = i

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True))
        self.L2 = nn.Sequential(
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())
        self.L3 = nn.Sequential(
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False))

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        n, c, t, h, w = x.size()
        n_batch = n
        new_x = x.contiguous()

        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t)
                                           
        local_activation = self.L(out.view(n_batch, c,
                                           t))
        local_activation2 = self.L2(local_activation).view(n_batch, c, t, 1, 1)
        local_activation3 = self.L3(local_activation).view(n_batch, c, t, 1, 1)

        new_x = new_x * local_activation2 + local_activation3 * 0.1 

        out = new_x.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 3, 4, 1).contiguous().reshape(n_batch, int(t*h*w), c)
        return out

class Block(nn.Module):

    def __init__(self,i,
                 img_size,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 cos_attn=False):
        super().__init__()
        self.img_size = img_size
        self.norm1 = norm_layer(dim)
        self.i = i
        self.tunning_mode='normal'
        if cos_attn:
            self.attn = CosAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        else:
            self.attn = Attention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
                attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.norm1.requires_grad_(False)
        self.attn.requires_grad_(False)
        self.drop_path.requires_grad_(False)
        self.norm2.requires_grad_(False)
        self.mlp.requires_grad_(False)
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) 
    
        if init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None
        
        # TAA module
        if self.i > 9:
            self.tam = TAA(self.i,in_channels=dim,
                        n_segment=8,
                        kernel_size=3,
                        stride=1,
                        padding=1)

    def forward(self, x):
        if self.gamma_1 is None:
            B,N,C = x.shape
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # x = x + self.drop_path(self.mlp(self.norm2(x)))  ## no TAA

            # TAA module in last two blocks
            if self.i > 9:
                y = self.norm2(x)

                # infer temporal segments and spatial grid from N and img_size (default patch_size=16)
                import math
                patch_size = 16
                # spatial patches per frame (e.g. (112/16)^2 = 49)
                sp_per_frame = (self.img_size // patch_size) ** 2
                if sp_per_frame <= 0:
                    raise RuntimeError(f"Invalid img_size {self.img_size} for patch_size {patch_size}")
                # temporal patch count
                t = int(N // sp_per_frame)
                if t * sp_per_frame != N:
                    # fallback: if N is a perfect square assume single temporal segment
                    s = int(round(math.sqrt(N)))
                    if s * s == N:
                        t = 1
                        h = s
                    else:
                        # final fallback: derive spatial grid from img_size
                        h = self.img_size // patch_size
                else:
                    h = int(math.sqrt(sp_per_frame))
                # reshape to (B, T, H, W, C) then permute to (B, C, T, H, W)
                y = y.view(B, t, h, h, C).permute(0, 4, 1, 2, 3)
                y = self.tam(y)
                # DO NOT call temporal_conv on a 5D tensor (Conv1d expects 3D input).
                # temporal_conv sẽ được áp dụng sau khi reshape về (B*H*W, C, T).
                x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path2(y)
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 num_frames=16,
                 tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))
        # self.proj.requires_grad_(False)

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.tensor(
        sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 head_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 with_cp=False,
                 cos_attn=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.depth=depth
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(
                num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(i,
                img_size=img_size,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)
        self.fc_norm = norm_layer(embed_dim)
        self.fc_norm2 = norm_layer(embed_dim)
        self.pos_embed.requires_grad_(False) 
        self.pos_drop.requires_grad_(False) 
        for name, param in self.blocks.named_parameters():
                if 'tam' in name:
                    # print(name)
                    pass
                else:
                    param.requires_grad = False
        self.norm.requires_grad_(False)
        self.patch_embed.requires_grad_(False)
        self.temporal_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, groups=embed_dim)


        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

        # self.head.weight.data.mul_(init_scale)
        # self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    # def get_classifier(self):
    #     return self.head

    # def reset_classifier(self, num_classes, global_pool=''):
    #     self.num_classes = num_classes
    #     self.head = nn.Linear(
    #         self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.size(0)

        x = self.patch_embed(x)
        if self.pos_embed is not None:
            B, N, C = x.shape
            # adapt positional embedding if sizes mismatch (common when temporal length differs)
            pe = self.pos_embed  # (1, P, C)
            if pe.shape[1] != N:
                import math, torch
                # try case: pe is spatial only (H*W) and new N == T * (H*W) -> repeat spatial pe across temporal dim
                P = pe.shape[1]
                s = int(round(math.sqrt(P))) if P > 0 else 0
                if s * s == P and N % P == 0:
                    T = N // P
                    # pe -> (1, H, W, C)
                    pe_sp = pe.view(1, s, s, C).permute(0, 3, 1, 2)  # 1,C,H,W
                    pe_sp = pe_sp.unsqueeze(2)  # 1,C,1,H,W
                    pe_rep = pe_sp.repeat(1, 1, T, 1, 1)  # 1,C,T,H,W
                    pe = pe_rep.permute(0, 2, 3, 4, 1).contiguous().view(1, N, C)
                else:
                    # fallback: try spatial interpolation when both are square grids (ignore temporal), else trim/repeat
                    try:
                        old_s = int(round(math.sqrt(P)))
                        new_s = int(round(math.sqrt(N)))
                        if old_s * old_s == P and new_s * new_s == N:
                            pe_map = pe.view(1, old_s, old_s, C).permute(0, 3, 1, 2)  # 1,C,old_s,old_s
                            pe_map = torch.nn.functional.interpolate(pe_map, size=(new_s, new_s), mode='bilinear', align_corners=False)
                            pe = pe_map.permute(0, 2, 3, 1).contiguous().view(1, N, C)
                        else:
                            # last resort: repeat or trim to match N
                            if P < N:
                                repeats = (N + P - 1) // P
                                pe = pe.repeat(1, repeats, 1)[:, :N, :].contiguous()
                            else:
                                pe = pe[:, :N, :].contiguous()
                    except Exception:
                        pe = pe.repeat(1, (N + P - 1) // P, 1)[:, :N, :].contiguous()
            x = x + pe.expand(B, -1, -1).type_as(x).to(x.device)
        x = self.pos_drop(x)

        for blk in self.blocks:
            if self.with_cp:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)
        

        if self.fc_norm is not None:

            B,N,C=x.shape
            # y = x.view(B,8,int(N/56),int(N/56),C).permute(0,4,1,2,3)  #112*112
            # Thay reshape cứng bằng reshape động:
            # B, N, C tính ở trên (N là số token không bao gồm cls nếu có)
            temporal_pref = getattr(self, 'num_segments', None)  # nếu model khởi tạo có num_segments thì ưu tiên
            found = None
            # nếu biết temporal mong muốn thì thử trước
            if temporal_pref is not None and temporal_pref > 0 and N % temporal_pref == 0:
                s = N // temporal_pref
                r = int(s**0.5)
                if r * r == s:
                    found = (temporal_pref, r)
            # tìm mọi cặp (T, side) sao cho T*side*side == N (ưu tiên T lớn hơn)
            if found is None:
                candidates = []
                for t in range(1, min(N, 64) + 1):  # giới hạn T đến 64 để nhanh
                    if N % t != 0:
                        continue
                    s = N // t
                    r = int(s**0.5)
                    if r * r == s:
                        candidates.append((t, r))
                if candidates:
                    # chọn temporal lớn nhất hợp lý (thường là nhiều frame hơn)
                    found = max(candidates, key=lambda x: x[0])
            if found is None:
                # debug giúp biết vì sao fail
                raise RuntimeError(f'Cannot reshape tokens: B={B}, N={N}, C={C}. '
                                   f'No (T, H, W) found such that T*H*W==N with H==W. '
                                   f'If you know temporal length, set self.num_segments accordingly.')
            T, side = found
            # reshape về (B, C, T, H, W)
            y = x.view(B, T, side, side, C).permute(0,4,1,2,3).contiguous()  # (B, C, T, H, W)
            # áp dụng Conv1d dọc trục temporal: đổi về (B*H*W, C, T)
            B, C, T, H, W = y.shape
            y = y.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, C, T)
            y = self.temporal_conv(y)  # (B*H*W, C, T2)
            T2 = y.size(-1)
            # trả về (B, C, T2, H, W)
            y = y.view(B, H, W, C, T2).permute(0, 3, 4, 1, 2).contiguous()
            # flatten về (B, C, H, W) để cộng với x (x có shape (B, N, C))
            feat_map = y.mean(dim=2)  # (B, C, H, W)
            return feat_map
             # return self.fc_norm((x.mean(1))), self.fc_norm(x)

        else:
            return self.norm(x[:, 0])

    def forward(self, x):
        # trả về trực tiếp feature map 4D (B, C, H, W)
        return self.forward_features(x)

# Add model factory functions expected by utils.py

@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    Minimal factory returning a VisionTransformer variant compatible with older code.
    Adjust embed_dim/depth/num_heads if you need exact original weights.
    """
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=0.0,
        use_learnable_pos_emb=False,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        **kwargs
    )
    if pretrained:
        # placeholder: user may load weights here
        pass
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depth=24,
        num_heads=8,
        mlp_ratio=4.0,
        **kwargs
    )
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224,
        patch_size=14,
        embed_dim=768,
        depth=32,
        num_heads=12,
        mlp_ratio=4.0,
        **kwargs
    )
    return model

@register_model
def vit_small_patch16_112(pretrained=False, img_size=112, **kwargs):
    """Small variant for 112x112 input (wrapper)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        **kwargs
    )

@register_model
def vit_base_patch16_112(pretrained=False, img_size=112, **kwargs):
    """Base variant for 112x112 input (wrapper)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        **kwargs
    )

@register_model
def vit_small_patch16_224(pretrained=False, img_size=224, **kwargs):
    """Small variant for 224x224 input (wrapper)."""
    return VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
        **kwargs
    )

# ensure attribute exists for imports
vit_small_patch16_224 = vit_small_patch16_224