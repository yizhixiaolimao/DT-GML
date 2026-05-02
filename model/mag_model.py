import torch
import torch.nn as nn

from model.fusion import FusionModel
from model.vit3d import ViT3D
from model.resnet import generate_model, FeatureFusionClassifier,TabularEmbedding_Robust
from model.dynamic_tanh import convert_ln_to_dyt
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F
def exists(val):
    return val is not None
def default(val, d):
    return val if exists(val) else d
class AftNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _ = x.shape
        h = self.heads
        context = default(context, x)

        # if kv_include_self:
        #     # cross attention requires CLS token includes itself as key / value
        #     context = torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class SIM_FeedForward_Updated(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim, num_vectors):
        super().__init__()
        self.aux_tabular = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(tabular_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, img_dim*4))
        )
        self.aux = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(9 * img_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, 2 * num_vectors))
        )
        self.scale_activation = nn.Sigmoid()
        self.reducion = nn.Sequential(nn.Conv1d(150, 4, 1),
                                      Rearrange('b n d -> b (d n)'))
        self.img_dim = img_dim
        self.num_vectors = num_vectors

    def forward(self, img, context):
        # global average pooling for image features
        context = torch.flatten(context, start_dim=1, end_dim=-1)
        img_feat = self.reducion(img)  # b, 16 * dim
        # feature transformation for tabular flattened features
        tabular_feat = self.aux_tabular(context) # context -> (b, num_tabular * 4)
        #进行L1标准化
        f_img = F.normalize(img_feat, dim=-1)
        f_text = F.normalize(tabular_feat, dim=-1)
        sim = F.cosine_similarity(f_img, f_text, dim=-1).unsqueeze(-1)
        out=torch.cat((img_feat, tabular_feat,sim), dim=-1)

        return out


class SIM_FeedForward(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim, num_vectors):
        super().__init__()
        self.proj_tab = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim)
        )
        self.affine = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, img_dim*2)
        )

        self.scale_activation = nn.Sigmoid()


    def forward(self, img, context):
        context_f = context.mean(dim=1)
        tabular_proj = self.proj_tab(context_f)
        f_img = F.normalize(img, dim=-1)  # (B, N, D)
        f_tab = F.normalize(tabular_proj, dim=-1)  # (B, D)
        f_tab = f_tab.unsqueeze(1).expand_as(f_img)
        tabular_feat = tabular_proj.view(tabular_proj.size()[0], 1, tabular_proj.size()[1]).expand_as(img)

        sim = (f_img * f_tab).sum(dim=-1, keepdim=True)  # (B, N, 1)

        affine_params = self.affine(sim)
        scale, shift = affine_params.chunk(2, dim=-1)
        scale=self.scale_activation(scale)

        out = img * scale + ((1 - scale) * tabular_feat) + shift

        return out

class IT_FeedForward_Updated(nn.Module):
    def __init__(self, img_dim, tabular_dim, hidden_dim, num_vectors):
        super().__init__()
        self.aux_tabular = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(tabular_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, img_dim))
        )
        self.aux = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(9 * img_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(hidden_dim, 2 * num_vectors))
        )
        self.scale_activation = nn.Sigmoid()
        self.reducion = nn.Sequential(nn.Conv1d(125, 8, 1),
                                      Rearrange('b n d -> b (d n)'))
        self.img_dim = img_dim
        self.num_vectors = num_vectors

    def forward(self, img, context):
        # global average pooling for image features
        squeeze = self.reducion(img)  # b, 16 * dim
        # feature transformation for tabular flattened features
        tabular_feat = self.aux_tabular(context) # context -> (b, num_tabular * 4)
        # get transformation parameters
        squeeze = torch.cat((squeeze, tabular_feat), dim=1)  # b, 2d
        attention = self.aux(squeeze)  # b, 2n
        v_scale, v_shift = torch.split(attention, self.num_vectors, dim=1) # b, n
        # expand to original img shape
        v_scale = v_scale.view(v_scale.size()[0], v_scale.size()[1], 1).expand_as(img)
        v_shift = v_shift.view(v_shift.size()[0], v_shift.size()[1], 1).expand_as(img)
        tabular_feat = tabular_feat.view(tabular_feat.size()[0], 1, tabular_feat.size()[1]).expand_as(img)
        # activate to [-1,1]
        v_scale = self.scale_activation(v_scale)
        # transform feature maps
        out = (v_scale * img) + v_shift
        return out

class Transformer_IT(nn.Module):
    def __init__(self, dim, tabular_dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.aux_tabular = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(tabular_dim, mlp_dim)),
            nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(mlp_dim, dim))
        )
        self.sim_fuse =SIM_FeedForward_Updated(dim, tabular_dim*6, mlp_dim, num_vectors=150)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, SIM_FeedForward(dim, tabular_dim, mlp_dim, num_vectors=150))
            ]))

    def forward(self, x, tabular, context=None):

        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x, context=tabular) + x
        x=self.norm(x)
        #这里是旧版的处理
        # tabular=self.aux_tabular(tabular)
        # f_img = F.normalize(x, dim=-1)
        # f_text = F.normalize(tabular, dim=-1)
        # #print('img:',f_img.shape)
        # #print('text:',f_text.shape)
        # f_img_mean=f_img.mean(dim=1)
        # f_text_mean=f_text.mean(dim=1)
        #
        # sim = F.cosine_similarity(f_img_mean, f_text_mean, dim=-1).unsqueeze(-1)
        # #print('sim:',sim.shape)
        # fused = torch.cat([f_img_mean, f_text_mean, sim], dim=-1)
        #新版处理
        fused = self.sim_fuse(x, context=tabular)

        return fused

class Model(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer_IT(dim=img_dim, tabular_dim=tab_dim*1, depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)
        self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)

        # 2. 融合
        fused_img = self.fuse1(img_feat,tab_feat)
        fused_tab = self.fuse2(tab_feat,img_feat)
        # print('img_fuse:', fused_img.shape)
        # print('tab_fuse:', fused_tab.shape)

        # 4. Decoder
        logits = self.decoder(fused_img,fused_tab)
        output = self.CLS_Head(logits)


        return output

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
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AftNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                AftNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3)
        )

    def forward(self, x,tab, context=None):
        for attn, ff in self.layers:
            x = attn(x, context=context) + x
            x = ff(x) + x
        x=self.norm(x)
        cls_head = x.mean(dim=1)
        #cls_tab = tab.mean(dim=1)
        #cls_fused = torch.cat([cls_head, cls_tab], dim=-1)


        output = self.to_latent(cls_head)
        cls_output = self.mlp_head(output)

        return cls_output

class Model_base(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_base, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        #self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        #self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer(dim=img_dim,depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)

        # self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)

        # 4. Decoder
        output = self.decoder(img_feat,tab_feat)



        return output


class Model_base_inter(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_base_inter, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer(dim=img_dim,depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)

        # self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        # print('img:',img_feat.shape)
        # print('tab:',tab_feat.shape)
        fused_img = self.fuse1(img_feat, tab_feat)
        fused_tab = self.fuse2(tab_feat, img_feat)


        # 4. Decoder
        output = self.decoder(fused_img,fused_tab)



        return output


class Model_base_inter_dyt(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_base_inter_dyt, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer(dim=img_dim,depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)

        self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        # print('img:',img_feat.shape)
        # print('tab:',tab_feat.shape)
        fused_img = self.fuse1(img_feat, tab_feat)
        fused_tab = self.fuse2(tab_feat, img_feat)


        # 4. Decoder
        output = self.decoder(fused_img,fused_tab)



        return output


class Model_base_inter_guide(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_base_inter_guide, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer_IT(dim=img_dim, tabular_dim=tab_dim*1, depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)
        #self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)

        # 2. 融合
        fused_img = self.fuse1(img_feat,tab_feat)
        fused_tab = self.fuse2(tab_feat,img_feat)
        # print('img_fuse:', fused_img.shape)
        # print('tab_fuse:', fused_tab.shape)

        # 4. Decoder
        logits = self.decoder(fused_img,fused_tab)
        output = self.CLS_Head(logits)


        return output


class Model_base_guide_dyt(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_base_guide_dyt, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer_IT(dim=img_dim, tabular_dim=tab_dim*1, depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)
        self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)



        # 4. Decoder
        logits = self.decoder(img_feat,tab_feat)
        output = self.CLS_Head(logits)


        return output


class Model_mri(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_mri, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        #self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        #self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer(dim=img_dim,depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)

        # self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(img_dim*8+1, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)

        # 4. Decoder
        output = self.decoder(img_feat,tab_feat)



        return output



class Model_tab(nn.Module):
    def __init__(self,
                 img_dim,
                 tab_dim,
                 hidden_dim,
                 tabular_continues_idx,
                 tabular_categorical_idx,
                 tabular_length,
                 seed=1037):
        super(Model_tab, self).__init__()
        # 1. CNN Encoder
        self.en1 = generate_model(18, out_dim=img_dim)
        # 3. Tabular Encoder
        self.en2 = TabularEmbedding_Robust(
            idx_real_features=tabular_continues_idx,
            idx_cat_features=tabular_categorical_idx,
            out_features=tab_dim,
            dropout_rate=0.5,
            hidden_units=[hidden_dim // 2, hidden_dim // 2]
        )

        # 4. Feature Fusion Layers
        #self.fuse1 = FeatureFusionClassifier(input_dim=tab_dim,output_dim=img_dim,input_batch=tabular_length,output_batch=150)
        #self.fuse2 = FeatureFusionClassifier(input_dim=img_dim,output_dim=tab_dim,input_batch=150,output_batch=tabular_length)

        # 5. Decoder / Final Head
        self.decoder =Transformer(dim=img_dim,depth=3,
                               heads=4, dim_head=img_dim // 4,
                               mlp_dim=img_dim * 4, dropout=0.1)

        # self.decoder = convert_ln_to_dyt(self.decoder)

        self.CLS_Head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4, 3)
        )

    def forward(self, img_input, tab_input):
        # img_input: 影像输入 (CNN)
        # tab_input: 表格输入

        # 1. 特征提取
        # img_feat = self.en1(img_input)
        tab_feat = self.en2(tab_input)
        # print('img:',img_feat.shape)
        print('tab:',tab_feat.shape)
        cls_tab = tab_feat.mean(dim=1)

        # 4. Decoder
        output = self.CLS_Head(cls_tab)



        return output