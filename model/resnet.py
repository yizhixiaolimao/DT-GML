import math
from functools import partial
from typing import Any, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def get_inplanes():
    return [32, 64, 128 ,256]

class ExU(nn.Module):
    """Exp-centered Unit activation"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # ExU(x) = x * exp(x.clamp(max=0))
        return x * torch.exp(torch.clamp(x, max=0))

class DeepCategoricalEmbedding(nn.Module):
    """
    Deep embedding for categorical features.
    Step 1: low-dim embedding lookup
    Step 2: MLP maps to high-dim embedding
    这里的n_values_list代表的是每一种类数据中包含的类别数
    """
    def __init__(self, n_values_list, low_dim=8, out_features=1024, mlp_hidden=[256,512]):
        """
        Args:
            n_values_list: list of int, number of possible values for each categorical feature
            low_dim: low-dimensional embedding for lookup table (step 1)
            out_features: output embedding dim (high-dim after MLP)
            mlp_hidden: list[int], hidden units for MLP
        """
        super().__init__()
        self.n_cat = len(n_values_list)
        self.low_dim = low_dim
        self.out_features = out_features

        # Step 1: low-dim embedding lookup table for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_values, low_dim) for num_values in n_values_list
        ])

        # Missing value representation for each feature
        self.missing_vectors = nn.ParameterList([
            nn.Parameter(torch.randn(low_dim)) for _ in n_values_list
        ])

        # Step 2: MLP maps low-dim embedding to high-dim
        mlp_layers = []
        in_dim = low_dim
        for hidden in mlp_hidden:
            mlp_layers.append(nn.Linear(in_dim, hidden))
            mlp_layers.append(nn.ReLU())
            in_dim = hidden
        mlp_layers.append(nn.Linear(in_dim, out_features))
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, val_categ: torch.LongTensor):
        """
        Args:
            val_categ: LongTensor of shape [batch, n_cat]
                        - each entry is the categorical index
                        - missing values can be indicated with -1
        Returns:
            features_categ: FloatTensor of shape [batch, n_cat, out_features]
        """
        batch_size = val_categ.size(0)
        features = []

        for j in range(self.n_cat):
            idx = val_categ[:, j]  # shape [batch]
            mask_missing = (idx == -1)
            idx_clamped = torch.clamp(idx, min=0)  # prevent -1 for embedding lookup

            # Step 1: lookup low-dim embedding
            emb_low = self.embeddings[j](idx_clamped)  # [batch, low_dim]

            # Replace missing values
            if mask_missing.any():
                emb_low[mask_missing] = self.missing_vectors[j]

            # Step 2: MLP maps to high-dim embedding
            emb_high = self.mlp(emb_low)  # [batch, out_features]
            features.append(emb_high)

        # Stack all categorical features along n_cat dimension
        features_categ = torch.stack(features, dim=1)  # [batch, n_cat, out_features]
        return features_categ

def conv3x3x3(in_planes, out_planes, stride=1):
        return nn.Conv3d(in_planes,
                         out_planes,
                         kernel_size=3,
                         stride=stride,
                         padding=1,
                         bias=False)



def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        #print("out:",out.shape)

        if self.downsample is not None:
            #print("or",residual.shape)
            residual = self.downsample(x)
            #print("after", residual.shape)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 out_dim,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        # self.conv0 = nn.Conv3d(1, n_input_channels, kernel_size=(conv1_t_size, 7, 7),
        #                        stride=(conv1_t_stride, 1, 1), padding=(conv1_t_size // 2, 3, 3), bias=False)

        #针对depth小的
        # self.conv1 = nn.Conv3d(n_input_channels,
        #                        self.in_planes,
        #                        kernel_size=(conv1_t_size, 7, 7),
        #                        stride=(conv1_t_stride, 2, 2),
        #                        padding=(conv1_t_size // 2, 3, 3),
        #                        bias=False)
        #针对三维相同的
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(2, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.reduce= nn.Sequential(
                nn.Conv3d(block_inplanes[3], out_dim, padding=0, kernel_size=1),
                nn.BatchNorm3d(out_dim),
                nn.ReLU()
            )

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_dim, n_classes)
        #self.fc = nn.Linear(1, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != (1, 1, 1) or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride=stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #x=self.conv0(x)
        x = self.conv1(x)
        #print("x:",x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        #print("l1:",x.shape)
        x = self.layer2(x)
        #print("l2:",x.shape)
        x = self.layer3(x)
        #print("l3:",x.shape)
        x = self.layer4(x)
        x = self.reduce(x)

        B, C, D, H, W = x.shape
        #print('x:',x.shape)
        feature = x.view(B, C, -1).permute(0,2,1)
        # token = x
        #
        # x = self.avgpool(feature)
        #
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return feature


def generate_model(model_depth,out_dim, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], out_dim ,get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], out_dim ,get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model



def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]

class FeatureFusionClassifier(nn.Module):
    def __init__(self, input_dim,output_dim,output_batch,input_batch,hidden_dim=256):
        super(FeatureFusionClassifier, self).__init__()
        # self.reducion = nn.Sequential(nn.Conv1d(input_batch, 8, 1),
        #                               Rearrange('b n d -> b (d n)'))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim*input_batch, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, 2*output_batch)
        )
        self.num_vectors=output_batch
        self.scale_activation = nn.Sigmoid()

    def forward(self, x1, x2):

        x2_flat = x2.flatten(start_dim=1)

        attention = self.mlp(x2_flat)
        v_scale, v_shift = torch.split(attention, self.num_vectors, dim=1)

        v_scale = v_scale.view(v_scale.size()[0], v_scale.size()[1], 1).expand_as(x1)
        v_shift = v_shift.view(v_shift.size()[0], v_shift.size()[1], 1).expand_as(x1)
        v_scale = self.scale_activation(v_scale)

        fused_features = v_scale * x1 + v_shift
        return fused_features

class DynamicTanh(nn.Module):
    def __init__(self, init_alpha: float = 0.5, init_beta: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, 0.01, 10.0)
        return self.beta * torch.tanh(alpha * x)

# ====== 主类：TabularEmbedding_Robust ======
class TabularEmbedding_Robust(nn.Module):
    def __init__(self,
                 idx_real_features: Sequence[int],
                 idx_cat_features: Sequence[int],
                 out_features: int,
                 hidden_units: Sequence[int],
                 dropout_rate: float = 0.5,
                 n_classes=2):
        super().__init__()
        self._idx_real_features = idx_real_features
        self._idx_cat_features = idx_cat_features
        self.out_features = out_features
        self.n_real_features = len(idx_real_features)

        # Step 1: gamma / beta for each real feature
        self.gamma = nn.Parameter(torch.ones(self.n_real_features, out_features))
        self.beta = nn.Parameter(torch.zeros(self.n_real_features, out_features))

        # Step 2: 共享 FFN
        layers = []
        in_features = out_features
        for hidden in hidden_units:
            layers.append(nn.utils.parametrizations.weight_norm(nn.Linear(in_features, hidden)))
            layers.append(nn.Dropout(dropout_rate))
            layers.append(DynamicTanh())
            in_features = hidden
        layers.append(nn.Linear(in_features, out_features))
        self.shared_ffn = nn.Sequential(*layers)

        # Pooling + 分类层
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_features, n_classes)

    # ====== Step 1 + Step 2 for real features ======
    def forward_real(self, x):
        """
        x: [B, n_real]
        return: [B, d, n_real]
        """
        #print("x:", x)
        B, n_real = x.shape  # B × n_real
        assert n_real == self.n_real_features, "输入的特征数量与初始化不符"

        # Step 1: Feature Expansion
        x = x.unsqueeze(-1)                       # [B, n_real, 1]
        x_expanded = x * self.gamma + self.beta   # [B, n_real, d]

        # Step 2: 共享 FFN
        # x_transformed = self.shared_ffn(x_expanded)  # [B, n_real, d]
        # x_out = x_transformed + x_expanded          # residual

        # [B, n_real, d] -> [B, d, n_real]
        x_out = x_expanded.permute(0, 2, 1)
        return x_out

    def forward(self, values: torch.Tensor):
        """
        values: [B, n_real]
        return:
          - token: [B, d, n_real]
          - cls_logits: [B, n_classes]
        """
        val_real = values[:, self._idx_real_features]  # 如果后续加cat，这里可以分开
        token = self.forward_real(val_real)       # [B, d, n_real]
        out =token.permute(0,2,1)

        # 分类分支
        # pooled = self.avgpool(token)                   # [B, d, 1]
        # pooled = pooled.view(pooled.size(0), -1)       # [B, d]
        # cls_logits = self.fc(pooled)

        return out
