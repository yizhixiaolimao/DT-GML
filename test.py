import torch
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve,cohen_kappa_score,classification_report
from pycm import ConfusionMatrix
import nibabel as nib
import pandas as pd
import os
from tqdm import tqdm
import csv

def safe_float(value, default=0.0):
    """确保指标可转为float，否则返回默认值"""
    try:
        if value is None or str(value).lower() == 'none' or value != value:  # 检测None和NaN
            return default
        return float(value)
    except Exception:
        return default
def cal_mcc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
def cross_entropy_loss(outputs, target_onehot):
    loss1 = -torch.sum(target_onehot * F.log_softmax(outputs, dim=1))
    return loss1
def sensitivity(y_true, y_pred):
    """
    y_true, y_pred: 0/1 数组
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
def precision(y_true, y_pred):
    """
    y_true, y_pred: 0/1 数组
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    return tp / (tp + fp) if (tp + fp) > 0 else 0.0
def evaluation(model, dataloader,device,fold,_class_=None):
    batch_size = 1
    model.eval()
    gt_list_sp= []
    pr_list_sp = []


    with torch.no_grad():
        for img, tab, label,_ in dataloader:
            img ,tab = img.to(device), tab.to(device)
            outputs = model(img,tab)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(probs, 1)
            out=predicted.to('cpu').detach().numpy()

            pr_list_sp.append(np.max(out))
            gt_list_sp.append(label.item())
        print(classification_report(gt_list_sp, pr_list_sp, labels=[0, 1 ,2],target_names=['class0', 'class1','class2']))
        # macro_f1 = round(f1_score(gt_list_sp, pr_list_sp, average='macro'),5)
        # acc = round(accuracy_score(gt_list_sp, pr_list_sp), 5)
        # # precis = round(precision(gt_list_sp, pr_list_sp), 5)
        # kappa = round(cohen_kappa_score(gt_list_sp, pr_list_sp),5)
        base_dir = "/home/cyf/TAB_IMG/MAG_guide"
        save_dir = os.path.join(base_dir, f"seed_{fold}")
        os.makedirs(save_dir, exist_ok=True)

        cm_val = ConfusionMatrix(actual_vector=gt_list_sp, predict_vector=pr_list_sp)
        # 宏平均指标
        Acc= safe_float(cm_val.Overall_ACC)
        Sen = safe_float(cm_val.TPR_Macro)
        F1 = safe_float(cm_val.F1_Macro)
        Mcc = safe_float(cm_val.Overall_MCC)
        Kappa = safe_float(cm_val.Kappa)
        Prec = safe_float(cm_val.PPV_Macro)
        print(
            f"ACC: {Acc:.5f}, SEN: {Sen:.5f}, F1: {F1:.5f}, MCC: {Mcc:.5f}, Kappa: {Kappa:.5f}, Precision: {Prec:.5f}")
        sen = {cls: safe_float(v) for cls, v in cm_val.TPR.items()}

        # 单类：Precision
        prec = {cls: safe_float(v) for cls, v in cm_val.PPV.items()}

        # 单类：F1
        f1 = {cls: safe_float(v) for cls, v in cm_val.F1.items()}

        # 单类：MCC
        mcc = {cls: safe_float(v) for cls, v in cm_val.MCC.items()}

        # acc
        class_acc = {}
        for cls in cm_val.classes:
            TP = cm_val.TP[cls]
            TN = cm_val.TN[cls]
            FP = cm_val.FP[cls]
            FN = cm_val.FN[cls]

            acc_cls = safe_float((TP + TN) / (TP + TN + FP + FN))
            class_acc[cls] = acc_cls

        print("Sensitivity (TPR):", sen)
        print("Precision (PPV):", prec)
        print("F1:", f1)
        print("MCC:", mcc)

        for cls in cm_val.classes:
            print(f"  Class {cls}:", class_acc[cls])

        # type for overall
        save_path = os.path.join(save_dir, "metric_overall.csv")
        metrics = {
            "Accuracy": safe_float(cm_val.Overall_ACC),
            "Sensitivity": safe_float(cm_val.TPR_Macro),
            "Precision": safe_float(cm_val.PPV_Macro),
            "F1-score": safe_float(cm_val.F1_Macro),
            "MCC": safe_float(cm_val.Overall_MCC),
            "Kappa": safe_float(cm_val.Kappa),
        }
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
        # type unique
        ovr_rows = []
        for cls in cm_val.classes:
            ovr_rows.append({
                "seed": fold,
                "class": cls,
                "Sensitivity": sen[cls],
                "Precision": prec[cls],
                "F1": f1[cls],
                "MCC": mcc[cls],
                "Accuracy": class_acc[cls],
            })
        save_path_ovr = os.path.join(save_dir, "metric_one_vs_rest.csv")
        df_ovr = pd.DataFrame(ovr_rows)
        df_ovr.to_csv(save_path_ovr, index=False)

    return Acc,F1,Kappa




def save_for_itk_snap(sensitivity_map, affine,id,output_dir='./itk_snap_results'):
    """
    为ITK-SNAP保存原始图像和热力图
    参数:
        original_mri: 原始MRI数据 (D, H, W)
        original_pet: 原始PET数据 (D, H, W)
        sensitivity_map: 热力图 (D, H, W)
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存热力图（轴顺序已转换）
    heatmap_path = os.path.join(output_dir, 'occlusion_heatmap',id)
    nib.save(nib.Nifti1Image(sensitivity_map, affine), heatmap_path)

    print(f"ITK-SNAP输入文件已保存至目录: {output_dir}")

    print(f"热力图路径: {heatmap_path}")

    return heatmap_path

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def compute_occlusion_sensitivity(model, input_3d,tab, window_size=(6, 6, 6), stride=6, target_class=1):
    """
    计算3D遮挡敏感性
    参数:
        model: 训练好的模型
        input_3d: 3D输入图像 (1, C, D, H, W) [单通道或多通道]
        window_size: 遮挡窗口大小
        stride: 滑动步长
        target_class: 要解释的目标类别
    返回:
        3D遮挡敏感性热力图
    """
    model.eval()
    B, C, D, H, W = input_3d.shape
    with torch.no_grad():
        original_output = model(input_3d,tab)
        original_prob = torch.softmax(original_output, dim=1)[0, target_class].item()

    # 初始化热力图
    sensitivity_map = torch.zeros(D, H, W, device=input_3d.device)

    # 计算单通道的均值（假设输入为单通道图像）
    input_mean = input_3d[0, 0].mean()  # 取第一个通道的均值

    # 遍历3D空间
    with torch.no_grad():
        for d in tqdm(range(0, D - window_size[0] + 1, stride), desc="Depth"):
            for h in range(0, H - window_size[1] + 1, stride):
                for w in range(0, W - window_size[2] + 1, stride):
                    # 创建遮挡后的输入
                    occluded_input = input_3d.clone()

                    # 遮挡单通道的对应区域
                    occluded_input[0, 0, d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] = input_mean

                    # 计算遮挡后的输出
                    occluded_output = model(occluded_input,tab)
                    occluded_prob = torch.softmax(occluded_output, dim=1)[0, target_class].item()

                    # 计算敏感性 (概率变化)
                    sensitivity = original_prob - occluded_prob

                    # 将敏感性值填充到对应区域
                    sensitivity_map[d:d + window_size[0], h:h + window_size[1], w:w + window_size[2]] += sensitivity

    # 归一化处理
    sensitivity_map = sensitivity_map / sensitivity_map.max()

    return sensitivity_map.cpu().numpy()


def occ_evaluation(model, dataloader,device,_class_=None):
    batch_size = 1
    model.eval()
    gt_list_sp= []
    pr_list_sp = []
    with torch.no_grad():
        for img, tab, label,path in dataloader:
            id = os.path.basename(path[0])
            img ,tab = img.to(device), tab.to(device)
            #计算敏感度图
            sensitivity_map = compute_occlusion_sensitivity(
                model,
                img,
                tab=tab,
                window_size=(8, 12, 8),
                stride=4,
                target_class=2  # 假设解释类别0
            )
            aimg = nib.load(path[0])
            affine = aimg.affine
            save_for_itk_snap(sensitivity_map, affine, id)



    return 0,0,0

