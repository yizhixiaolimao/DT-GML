import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
import random
from torch import nn
import math
MIN_NUM_PATCHES = 16
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1037)  # 任意固定值

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,num_batches=2048,dtype=torch.float):
        super().__init__()
        inner_dim = dim_head * heads
        self.num_heads = heads
        dd_std = 0.05 * math.sqrt(2 / ( dim_head+ dim+1))
        self.heads = heads
        self.scale = dim ** -0.5
        self.dw_activation = nn.Tanh()

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        #b为1，n为1024+1,d为2048
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        #此时dim为512

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        #mask操作
        mask_value = -torch.finfo(dots.dtype).max
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        #print("attn shape:", attn.shape)


        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,num_batches):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads,
                                                dim_head=dim_head, dropout=dropout,num_batches=num_batches))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        #("x:",x.shape)
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT3D(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,num_batches, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        num_patches = num_batches
        patch_dim = patch_size ** 3
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        #self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout,num_batches)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        #self.apply(self._init_weights)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
        #         if m.bias is not None:
        #             nn.init.zeros_(m.bias)
        #     elif isinstance(m, nn.Parameter):
        #         if m.shape[-1] == self.dim:
        #             nn.init.trunc_normal_(m, std=0.02)

    def _init_weights(self, module):
        """ Initialize weights for different modules """
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            # 针对pos_embedding和cls_token
            if module.dim() == 2 and module.shape[-1] == self.dim:
                nn.init.trunc_normal_(module, std=0.02)
    def forward(self, img, mask=None):
        # p = self.patch_size
        # b, c, d, h, w=img.shape
        #
        # x = rearrange(
        #     img, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=p, p2=p, p3=p)
        # print('transx',img.shape)
        x = img.view(img.size(0), img.size(1), -1)
        #x = rearrange(img, 'b c d h w -> b c (d h w)')
        #print('x:',x.shape)
        #x = self.patch_to_embedding(x)
        #print("x:",x.shape)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x=self.mlp_head(x)
        return x

if __name__ == '__main__':
    a = torch.randn(1,2048, 12, 12,12)
    decoder = ViT3D(
        image_size=(12, 12, 12),
        patch_size=12,
        num_classes=2,
        dim=1024,  # token dim
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.2,
        emb_dropout=0.2,
        num_batches=2048
    )
    output=decoder(a)
    print('output',output)

