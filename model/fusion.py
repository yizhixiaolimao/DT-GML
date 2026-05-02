import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class FusionModel(nn.Module):
    def __init__(self, img_dim, tab_dim, fuse_dim, num_classes, seed=42):
        super(FusionModel, self).__init__()
        self.img_dim = img_dim
        self.tab_dim = tab_dim
        self.fuse_dim = fuse_dim
        self.num_classes = num_classes

        # 固定随机种子（方便复现）
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 如果 img_dim != tab_dim，就映射到同一维度
        if img_dim != tab_dim:
            self.img_proj = nn.Linear(img_dim, fuse_dim // 4)
            self.tab_proj = nn.Linear(tab_dim, fuse_dim // 4)
        else:
            self.img_proj = nn.Identity()
            self.tab_proj = nn.Identity()

        # MLP 分类头
        self.classifier = nn.Sequential(
            nn.Linear(fuse_dim*4, fuse_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fuse_dim // 2, num_classes)
        )

    def forward(self, img, tab):
        """
        img: (B, N, D)
        tab: (B, N, D)
        """
        # --- 对齐维度 ---
        img = self.img_proj(img)
        tab = self.tab_proj(tab)

        # --- Global Average & Max Pooling ---
        img_gap = img.mean(dim=1)             # (B, D)
        img_gmp = img.amax(dim=1)             # (B, D)
        tab_gap = tab.mean(dim=1)             # (B, D)
        tab_gmp = tab.amax(dim=1)             # (B, D)

        # --- 拼接 ---
        fused = torch.cat([img_gap, img_gmp, tab_gap, tab_gmp], dim=1)  # (B, 4*D)

        # --- 分类 ---
        out = self.classifier(fused)  # (B, num_classes)
        return out

