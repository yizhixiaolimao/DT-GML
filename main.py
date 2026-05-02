import torch
from torch.utils.data.sampler import WeightedRandomSampler
from data.dataset import get_data_transforms,MedicalDataset,PTIDMRIDataset
import numpy as np
import random
import monai
import os
from model.mag_model import Model
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from test import evaluation
from torch.nn import functional as F
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts

torch.autograd.set_detect_anomaly(True)

def cross_entropy_loss(outputs, target_onehot):
    log_probs = F.log_softmax(outputs, dim=1)
    loss = -torch.mean(torch.sum(target_onehot * log_probs, dim=1))
    return loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_seed(seed):
    monai.utils.set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    epochs = 120

    batch_size = 2
    image_size = 96

    #表格数据的标签，待修改,目前有6个数据，年龄及五个评估指标
    tabular_continues_idx=[0,1,2,3,4,5]
    tabular_categorical_idx=None
    img_dim=64
    tab_dim=4
    hidden_dim=64

        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, test_transform = get_data_transforms(image_size, image_size)
    fold=1037
    csv_path = f"/home/cyf/TAB_IMG/MAG_cls/csv_data/all/{fold}"
    img_path = "/data/birth/cyf/shared_data/CAPL/Tab_img/ALL"
    ckp_path = f"/data/birth/cyf/output/CAPL/TAB_img/{fold}/last_model.pth"
    ckp_path_best = f"/data/birth/cyf/output/CAPL/TAB_img/{fold}/best_model.pth"
    os.makedirs(os.path.dirname(ckp_path), exist_ok=True)
    #train_data = MedicalDataset(root=train_path, transform=None)

    train_data = PTIDMRIDataset(root=img_path, path=csv_path,transform=test_transform,phase="train.csv")
    test_data = PTIDMRIDataset(root=img_path, path=csv_path,transform=test_transform,phase="test.csv")
    print("训练集数量:", len(train_data))
    print("测试集数量:", len(test_data))
    #weights sampler
    weights=train_data.get_sample_weights()
    #print('weights',weights)
    #sampler = WeightedRandomSampler(weights, len(weights))


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    model = Model(
        img_dim=img_dim,
        tab_dim=tab_dim,
        hidden_dim=hidden_dim,
        tabular_continues_idx=tabular_continues_idx,
        tabular_categorical_idx=tabular_categorical_idx,
        tabular_length=len(tabular_continues_idx)
    )
    #编码器1
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-4,amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-5)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=epochs//2,  # 首次周期长度（单位 epoch）
    #     T_mult=2,  # 每次重启后周期翻倍
    #     eta_min=1e-5
    # )

    best_val_acc = float('-inf')  # 跟踪验证集最佳准确率
    best_acc = float('-inf')  # 跟踪测试集最佳准确率（仅在验证集表现好时更新）
    model.to(device)
    for epoch in range(epochs):
        model.train()
        loss_list = []
        min_loss=float('inf')
        for i,(img , tab , label,_) in enumerate(train_dataloader):
            img ,tab, label = img.to(device), tab.to(device),label.to(device)
            batch_size=img.size(0)

            target_onehot3 = F.one_hot(label, num_classes=3).float().to(device)
            optimizer.zero_grad()
            output=model(img,tab)

            loss = cross_entropy_loss(output, target_onehot3)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step()
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        if (epoch + 1) % 10 == 0:
            acc, f1, kappa = evaluation(model, test_dataloader, device, fold)
            print('Validation - acc:{:.4f}, f1{:.4f}, kappa{:.4f}'.format(acc, f1, kappa))
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save({'model': model.state_dict(), "optim": optimizer.state_dict()}, ckp_path_best)
                print("best_val_acc:", best_val_acc, "epoch:", epoch)
            torch.save({'model': model.state_dict(),"optim":optimizer.state_dict()}, ckp_path)
    #最终测试
    print('Finished Training')
    checkpoint = torch.load(ckp_path_best)
    model.load_state_dict(checkpoint['model'])
    print('----Testing----')
    acc, f1, kappa = evaluation(model, test_dataloader, device, fold)
    print('Validation - acc:{:.4f}, f1{:.4f}, kappa{:.4f}'.format(acc, f1, kappa))
    return acc, f1, kappa




if __name__ == '__main__':
    seed=1037
    setup_seed(seed)
    train()

