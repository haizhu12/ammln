import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_model import vit_base_patch16_224_in21k as create_model
from functools import partial
from collections import OrderedDict


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        k_size = 7
        patch_size=img_size/k_size
        patch_size=int(patch_size)
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        #self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.grid_size = (img_size[0] // k_size, img_size[1] // k_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=k_size, stride=k_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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



class SimAM(nn.Module):
    def __init__(self, eps=1e-4):
        super(SimAM, self).__init__()
        self.eps = eps

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w
        d = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = d / (4 * (d.sum(dim=[2, 3], keepdim=True) / n + self.eps)) + 0.5
        return torch.sigmoid(y)


class EnergyAttention(nn.Module):
    def __init__(self, low_dim, high_dim):
        super(EnergyAttention, self).__init__()
        self.conv = nn.Conv2d(high_dim, low_dim, kernel_size=3, padding=1)
        self.atte = SimAM()

    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(self.conv(high_feat), low_feat.size()[-2:], mode='bilinear', align_corners=False)
        sdsd=torch.relu(low_feat + high_feat)
        atte = self.atte(torch.relu(low_feat + high_feat))
        low_feat = atte * low_feat
        return atte, low_feat


class Model(nn.Module):
    def __init__(self, backbone_type, proj_dim,num_classes=1000,img_size=224, patch_size=16, in_c=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., norm_layer=None,
                 act_layer=None):
        super(Model, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        #self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        self.patch_embed0 = PatchEmbed(img_size=56, patch_size=patch_size, in_c=256, embed_dim=embed_dim)
        self.patch_embed = PatchEmbed(img_size=28, patch_size=patch_size, in_c=512, embed_dim=embed_dim)
        self.patch_embed2 = PatchEmbed(img_size=14, patch_size=7, in_c=1024, embed_dim=embed_dim)
        self.patch_embed3 = PatchEmbed(img_size=7, patch_size=4, in_c=2048, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches
        # num_patches2 = self.patch_embed2.num_patches
        # num_patches3 = self.patch_embed3.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, num_patches2 + self.num_tokens, embed_dim))
        # self.pos_embed3 = nn.Parameter(torch.zeros(1, num_patches3 + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)


        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        #nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        # backbone
        self.backbone = timm.create_model('seresnet50' if backbone_type == 'resnet50' else 'vgg16_bn',
                                          features_only=True, out_indices=(1,2, 3, 4), pretrained=True)
        dims = [512, 1024, 2048] if backbone_type == 'resnet50' else [256, 512, 512]

        # atte
        self.energy_1 = EnergyAttention(dims[0], dims[2])
        self.energy_2 = EnergyAttention(dims[1], dims[2])
        self.energy_4 = EnergyAttention(256, dims[2])
        #self.energy_4 = EnergyAttention(dims[0], dims[1],dims[2])
        # proj
        #self.proj = nn.Linear(sum(dims)+3*embed_dim, proj_dim)
        self.proj = nn.Linear(sum(dims)+embed_dim, proj_dim)
        #num_class
        self.linear = nn.Linear(in_features=proj_dim, out_features=num_classes, bias=False)
        #self.linear1 = nn.Linear(sum(dims)+dims[0], num_classes, bias=False)
    def forward(self, img):
        block_0_feat,block_1_feat, block_2_feat, block_3_feat = self.backbone(img)
        #x = self.forward_features(img)
        #x0=self.forward_Atten0(block_0_feat)
        x1=self.forward_Atten(block_1_feat)
        x2=self.forward_Atten2(block_2_feat)
        x3=self.forward_Atten3(block_3_feat)
        x=x1+x2+x3
        #x=torch.sigmoid(x)
        # block_4_atte, block_4_feat = self.energy_4(block_1_feat, block_2_feat)
        # block_4_atte, block_4_feat = self.energy_1(block_4_feat, block_3_feat)
        # block_1_atte, block_1_feat = self.energy_1(block_1_feat, block_3_feat)
        # block_2_atte, block_2_feat = self.energy_2(block_2_feat, block_3_feat)

        #block_3_atte = torch.sigmoid(block_3_feat)
        #block_0_feat = torch.flatten(F.adaptive_max_pool2d(block_0_feat, (1, 1)), start_dim=1)
        block_1_feat = torch.flatten(F.adaptive_max_pool2d(block_1_feat, (1, 1)), start_dim=1)
        block_2_feat = torch.flatten(F.adaptive_max_pool2d(block_2_feat, (1, 1)), start_dim=1)
        block_3_feat = torch.flatten(F.adaptive_max_pool2d(block_3_feat, (1, 1)), start_dim=1)
        #block_4_feat = torch.flatten(F.adaptive_max_pool2d(block_4_feat, (1, 1)), start_dim=1)
        #feat = torch.cat((block_1_feat, block_2_feat, block_3_feat,x1,x2,x3), dim=-1)
        feat = torch.cat((block_1_feat, block_2_feat, block_3_feat,x), dim=-1)
        proj = self.proj(feat)
        #ouput_kd=self.linear1(feat)
        out=self.linear(proj)
        #return block_1_atte, block_2_atte, block_3_atte, out, F.normalize(proj, dim=-1)
        return block_1_feat, block_1_feat, block_1_feat, out, F.normalize(proj, dim=-1)

    # def forward_features(self, x):
    #     # [B, C, H, W] -> [B, num_patches, embed_dim]
    #     x = self.patch_embed(x)  # [B, 196, 768]
    #     # [1, 1, 768] -> [B, 1, 768]
    #     cls_token = self.cls_token.expand(x.shape[0], -1, -1)
    #     if self.dist_token is None:
    #         x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
    #     else:
    #         x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
    #
    #     x = self.pos_drop(x + self.pos_embed)
    #     x = self.blocks(x)
    #     x = self.norm(x)
    #     if self.dist_token is None:
    #         return self.pre_logits(x[:, 0])
    #     else:
    #         return x[:, 0], x[:, 1]

    def forward_Atten0(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]

        x = self.patch_embed0(x)  # [B, 196, 768]

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        #x = self.pos_drop(x + self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)

        x = self.norm(x)
        x = torch.sigmoid(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward_Atten(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]

        x = self.patch_embed(x)  # [B, 196, 768]

        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        #x = self.pos_drop(x + self.pos_embed)
        x = self.pos_drop(x)
        x = self.blocks(x)

        x = self.norm(x)
        x = torch.sigmoid(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward_Atten2(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed2(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)

        x = self.norm(x)
        x = torch.sigmoid(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]
    def forward_Atten3(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed3(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x)
        x = self.blocks(x)

        x = self.norm(x)
        x = torch.sigmoid(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]