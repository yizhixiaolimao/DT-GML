from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np
import nibabel as nib
import skimage.transform
import monai
monai.utils.set_determinism(seed=1037)
from monai.transforms import (Compose, ScaleIntensity, CenterSpatialCrop,
                              RandAdjustContrast, RandRotate, RandZoom, RandFlip,
                              RandAdjustContrastd, RandRotated, RandZoomd, RandFlipd,ToTensor)
# def standardize_3d_slicewise(vol):
#     #print("vol", vol.shape)  (256,256,64)
#     standardized_vol = np.zeros_like(vol)
#     for z in range(vol.shape[2]):  # 假设 vol 形状为 [Depth, Height, Width]
#         slice = vol[:,:,z]
#         mean = slice.mean()
#         std = slice.std()
#         standardized_vol[:,:,z] = (slice - mean) / (std + 1e-8)  # 防止除零
#     return standardized_vol

def standardize_3d_slicewise(vol):
    mean = np.mean(vol, axis=(0, 1), keepdims=True)  # 形状 (1, 1, 64)
    std = np.std(vol, axis=(0, 1), keepdims=True)    # 形状 (1, 1, 64)
    return (vol - mean) / (std + 1e-8)

def is_empty_slice(slice_data):
    """判断一个切片是否全为0"""
    return np.all(slice_data == 0)


def resize_data_z_axis_none(data, img, target_spacing=2.5):
    """
    根据目标间距（target_spacing）调整数据在z轴上的分辨率。

    参数:
        data (numpy.ndarray): 输入的3D数据，形状为 (x, y, z)。
        img (nibabel.Nifti1Image): 包含原始图像信息的NIfTI对象。
        target_spacing (float): 目标z轴间距（默认为2.5）。

    返回:
        numpy.ndarray: z轴调整后的数据，形状为 (x, y, new_z)。
    """
    original_shape = data.shape
    #print("ori",original_shape)
    zooms = img.header.get_zooms()
    z_spacing = zooms[2]  # 假设第三个维度是z轴

    # 计算新的z轴大小
    z_original = original_shape[2]
    new_z = int(np.round(z_original * z_spacing / target_spacing))

    # 调整z轴分辨率
    data_z_aligned = skimage.transform.resize(
        data,
        (original_shape[0], original_shape[1],new_z),
        order=1,  # 线性插值
        anti_aliasing=True,
        preserve_range=True
    )
    current_z = data_z_aligned.shape[2]

    if current_z > 80:
        # 检查顶部第一层是否全0
        if is_empty_slice(data_z_aligned[:, :, -1]):  # 顶部第一层
            # 如果顶部第一层是全0，从顶部开始裁剪全0层
            top_crop_index = 0
            for i in range(current_z - 1, -1, -1):  # 从顶部向底部遍历
                if not is_empty_slice(data_z_aligned[:, :, i]):
                    top_crop_index = i + 1  # 找到第一个非全0层，记录其下一层索引
                    break

            # 裁剪上方全0层
            data_cropped_top = data_z_aligned[:, :, :top_crop_index]

            # 如果裁剪后仍然大于80，从底部向上裁剪
            if data_cropped_top.shape[2] > 80:
                data_processed = data_cropped_top[:, :, -80:]
            else:
                data_processed = data_cropped_top
        else:
            # 如果顶部第一层不是全0，直接从底部裁剪到80层
            data_processed = data_z_aligned[:, :, -80:]
    else:
        # 如果z轴尺寸小于80，直接padding
        pad_total = 64 - current_z
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        data_processed = np.pad(
            data_z_aligned,
            pad_width=((0, 0), (0, 0), (pad_top, pad_bottom)),
            mode='constant',
            constant_values=0
        )

    #归一化处理
    min_val = np.min(data_processed)
    max_val = np.max(data_processed)
    if max_val != min_val:
        data_normalized = (data_processed - min_val) / (max_val - min_val)
    else:
        data_normalized = np.zeros_like(data_processed)

    # mask = data_processed > 0
    # mu = np.mean(data_processed[mask])
    # sigma = np.std(data_processed[mask])
    # data_normalized = np.zeros_like(data_processed)
    # data_normalized[mask] = (data_processed[mask] - mu) / sigma



    return data_normalized
def resize_data_z_axis(data, img, target_spacing=1.5):
    """
    根据目标间距（target_spacing）调整数据在z轴上的分辨率。

    参数:
        data (numpy.ndarray): 输入的3D数据，形状为 (x, y, z)。
        img (nibabel.Nifti1Image): 包含原始图像信息的NIfTI对象。
        target_spacing (float): 目标z轴间距（默认为2.5）。

    返回:
        numpy.ndarray: z轴调整后的数据，形状为 (x, y, new_z)。
    """
    original_shape = data.shape
    # print("ori",original_shape)
    zooms = img.header.get_zooms()
    z_spacing = zooms[2]  # 假设第三个维度是z轴

    # 计算新的z轴大小
    z_original = original_shape[2]
    new_z = int(np.round(z_original * z_spacing / target_spacing))

    # 调整z轴分辨率
    data_z_aligned = skimage.transform.resize(
        data,
        (original_shape[0], original_shape[1],new_z),
        order=1,  # 线性插值
        anti_aliasing=True,
        preserve_range=True
    )
    current_x = data_z_aligned.shape[0]
    current_y = data_z_aligned.shape[1]
    current_z = data_z_aligned.shape[2]

    if current_x < 96:
        # 如果x轴尺寸小于96，直接padding
        pad_total = 96 - current_x
        pad_right = pad_total // 2
        pad_left = pad_total - pad_right
        data_processed_x = np.pad(
            data_z_aligned,
            pad_width=((pad_left, pad_right), (0, 0), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        data_processed_x = data_z_aligned
    if current_y < 96:
        # 如果y轴尺寸小于96，直接padding
        pad_total = 96 - current_y
        pad_front = pad_total // 2
        pad_back = pad_total - pad_front
        data_processed_y = np.pad(
            data_processed_x,
            pad_width=((0, 0), (pad_back, pad_front), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        data_processed_y = data_processed_x

    #处理z
    if current_z > 96:
        # 检查顶部第一层是否全0
        if is_empty_slice(data_processed_y[:, :, -1]):  # 顶部第一层
            # 如果顶部第一层是全0，从顶部开始裁剪全0层
            top_crop_index = 0
            for i in range(current_z - 1, -1, -1):  # 从顶部向底部遍历
                if not is_empty_slice(data_processed_y[:, :, i]):
                    top_crop_index = i + 1  # 找到第一个非全0层，记录其下一层索引
                    break

            # 裁剪上方全0层
            data_cropped_top = data_processed_y[:, :, :top_crop_index]

            # 如果裁剪后仍然大于80，从底部向上裁剪
            if data_cropped_top.shape[2] > 96:
                data_processed = data_cropped_top[:, :, -96:]
            else:
                data_processed = data_cropped_top
        else:
            # 如果顶部第一层不是全0，直接从底部裁剪到96层
            data_processed = data_processed_y[:, :, -96:]
    else:
        # 如果z轴尺寸小于96，直接padding
        pad_total = 96 - current_z
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        data_processed = np.pad(
            data_processed_y,
            pad_width=((0, 0), (0, 0), (pad_top, pad_bottom)),
            mode='constant',
            constant_values=0
        )

    #归一化处理
    min_val = np.min(data_processed)
    max_val = np.max(data_processed)
    if max_val != min_val:
        data_normalized = (data_processed - min_val) / (max_val - min_val)
    else:
        data_normalized = np.zeros_like(data_processed)

    # mask = data_processed > 0
    # mu = np.mean(data_processed[mask])
    # sigma = np.std(data_processed[mask])
    # data_normalized = np.zeros_like(data_processed)
    # data_normalized[mask] = (data_processed[mask] - mu) / sigma


    # print('after:',data_normalized.shape)
    return data_normalized


class CenterCrop3D(object):
    def __init__(self, crop_size):
        assert isinstance(crop_size, (tuple, list)) and len(crop_size) == 3, "crop_size should be a tuple of length 3"
        self.crop_size = crop_size
        self.target_size=crop_size

    def __call__(self, sample):
        """
        Args:
            sample (numpy.ndarray or torch.Tensor): 3D image with shape (D, H, W)
        Returns:
            Cropped/padded 3D image with target size (96, 96, 96)
        """
        d, h, w = sample.shape[0], sample.shape[1], sample.shape[2]
        t_d, t_h, t_w = self.target_size

        # 先处理深度维度(D)
        if d < t_d:
            # Padding
            pad_d = (t_d - d) // 2
            extra_d = (t_d - d) % 2
            padding_d = (pad_d, pad_d + extra_d)
        else:
            # Cropping
            start_d = (d - t_d) // 2
            padding_d = (0, 0)
            sample = sample[start_d:start_d + t_d, :, :]

        # 处理高度维度(H)
        if h < t_h:
            pad_h = (t_h - h) // 2
            extra_h = (t_h - h) % 2
            padding_h = (pad_h, pad_h + extra_h)
        else:
            start_h = (h - t_h) // 2
            padding_h = (0, 0)
            sample = sample[:, start_h:start_h + t_h, :]

        # 处理宽度维度(W)
        if w < t_w:
            pad_w = (t_w - w) // 2
            extra_w = (t_w - w) % 2
            padding_w = (pad_w, pad_w + extra_w)
        else:
            start_w = (w - t_w) // 2
            padding_w = (0, 0)
            sample = sample[:, :, start_w:start_w + t_w]

        # 如果有任一维度需要padding
        if any(p > 0 for p in padding_d + padding_h + padding_w):
            if isinstance(sample, torch.Tensor):
                # 对PyTorch Tensor进行padding
                padding = padding_w + padding_h + padding_d  # PyTorch的padding顺序是反的
                sample = torch.nn.functional.pad(sample, padding)
            else:
                # 对numpy数组进行padding
                padding = [(padding_d[0], padding_d[1]),
                          (padding_h[0], padding_h[1]),
                          (padding_w[0], padding_w[1])]
                sample = np.pad(sample, padding, mode='constant')

        return sample

def get_data_transforms(size, isize):
    data_transforms = Compose([
        ScaleIntensity(),
        CenterSpatialCrop([160, 192, 160]),
        RandAdjustContrast(prob=0.5),
        RandRotate(prob=0.2, range_x=0.05),
        RandZoom(prob=0.2, min_zoom=0.95),
        ToTensor(),
    ])
    gt_transforms = Compose([
        ScaleIntensity(),
        CenterSpatialCrop([160, 192, 160]),
        ToTensor()
    ])
    return data_transforms, gt_transforms



class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, img_type

def load_data(dataset_name='mnist',normal_class=0,batch_size='16'):

    if dataset_name == 'cifar10':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            #transforms.CenterCrop(28),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        os.makedirs("./Dataset/CIFAR10/train", exist_ok=True)
        dataset = CIFAR10('./Dataset/CIFAR10/train', train=True, download=True, transform=img_transform)
        print("Cifar10 DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/CIFAR10/test", exist_ok=True)
        test_set = CIFAR10("./Dataset/CIFAR10/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'mnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/MNIST/train", exist_ok=True)
        dataset = MNIST('./Dataset/MNIST/train', train=True, download=True, transform=img_transform)
        print("MNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/MNIST/test", exist_ok=True)
        test_set = MNIST("./Dataset/MNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)

    elif dataset_name == 'fashionmnist':
        img_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])

        os.makedirs("./Dataset/FashionMNIST/train", exist_ok=True)
        dataset = FashionMNIST('./Dataset/FashionMNIST/train', train=True, download=True, transform=img_transform)
        print("FashionMNIST DataLoader Called...")
        print("All Train Data: ", dataset.data.shape)
        dataset.data = dataset.data[np.array(dataset.targets) == normal_class]
        dataset.targets = [normal_class] * dataset.data.shape[0]
        print("Normal Train Data: ", dataset.data.shape)

        os.makedirs("./Dataset/FashionMNIST/test", exist_ok=True)
        test_set = FashionMNIST("./Dataset/FashionMNIST/test", train=False, download=True, transform=img_transform)
        print("Test Train Data:", test_set.data.shape)


    elif dataset_name == 'retina':
        data_path = 'Dataset/OCT2017/train'

        orig_transform = transforms.Compose([
            transforms.Resize([128, 128]),
            transforms.ToTensor()
        ])

        dataset = ImageFolder(root=data_path, transform=orig_transform)

        test_data_path = 'Dataset/OCT2017/test'
        test_set = ImageFolder(root=test_data_path, transform=orig_transform)

    else:
        raise Exception(
            "You enter {} as dataset, which is not a valid dataset for this repository!".format(dataset_name))

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
    )

    return train_dataloader, test_dataloader


class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root,root_csv, transform, phase):

        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        if phase == 'train':
            self.pet_path = os.path.join(root_csv, 'train')
        else:
            self.pet_path = os.path.join(root_csv, 'test')
        self.transform = transform
        self.phase = phase


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path,label = self.img_paths[idx],self.labels[idx]
        pet_path = img_path.replace("/mri/", "/pet/")
        # img_path = glob.glob(os.path.join(img_path, "nii", "*.*"))[0]
        # pet_path = glob.glob(os.path.join(pet_path, "nii", "*.*"))[0]
        id=os.path.basename(img_path)
        img = nib.load(img_path)
        pet = nib.load(pet_path)
        data=img.get_fdata()  #(256,256,166)
        pet_data=pet.get_fdata()  #(160,160,96)
        # print('data',data.shape)
        # print('pet',pet_data.shape)

        #对齐体素
        data_z_aligned=resize_data_z_axis(data, img, target_spacing=1.5)
        pet_z_aligned=resize_data_z_axis(pet_data, pet, target_spacing=1.5)

        # data1 = self.transform(data.astype(np.float32()))
        # pet_data1 = self.transform(pet_data.astype(np.float32()))

        data1=self.transform(data_z_aligned.astype(np.float32()))
        pet_data1=self.transform(pet_z_aligned.astype(np.float32()))
        # print("after:",pet_data.shape)

        data1=data1[None,...]
        pet_data1=pet_data1[None,...]
        # data = np.load(img_path)
        # data = np.expand_dims(data, axis=0)
        #img = Image.fromarray(data)
        # print("data:",data.size)

        #img = Image.open(img_path).convert('RGB')
        #img = self.transform(img)

        return data1, pet_data1,label,img_path


from typing import Optional, Callable, Union, Tuple, Dict
import pandas as pd



class PTIDMRIDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root:str,
            path: str,
            phase: str,
            ptid_col: str = "ptid",
            mri_col: str = "image id",
            label_col: str="type",
            deduplicate: bool = False,
            keep: Union[str, bool] = "first",
            return_dict: bool = False,
            transform: Optional[Callable[[Union[Tuple[str, str], Dict[str, str]]],
            Union[Tuple, Dict]]] = None,
    ):
        csv_path=os.path.join(path,phase)
        df = pd.read_csv(csv_path)

        # --- 列名自动识别（大小写不敏感 + 常见别名/模糊匹配） ---
        cols = list(df.columns)

        # 可选：按 PTID 去重（保留首次出现，保持原始顺序）
        if deduplicate:
            df = df.drop_duplicates(subset=[ptid_col], keep=keep)

        self.path=root
        self.df = df.reset_index(drop=True)
        self.ptid_col = ptid_col
        self.mri_col = mri_col
        self.root = root
        self.label_col = label_col
        self.return_dict = return_dict
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[str, str], Dict[str, str]]:

        row = self.df.iloc[idx]
        # 转为字符串并处理缺失
        ptid = None if pd.isna(row[self.ptid_col]) else str(row[self.ptid_col])
        mri_id = None if pd.isna(row[self.mri_col]) else str(row[self.mri_col]).replace(",", "")

        label_id = None if pd.isna(row[self.label_col]) else str(row[self.label_col])
        #/data/birth/cyf/shared_data/CAPL/normalized_to_mni152/002_S_0295_I238627_MRI_brain.nii.gz
        if label_id =='CN':
            label=0
        elif label_id =='SMC':
            label=1
        elif label_id =='MCI' or label_id=='LMCI' or label_id=='EMCI':
            label=2
        else:
            print('wrong')
        name = f"{ptid}_{mri_id}_MRI_brain.nii.gz"
        #这里是根据id得到name，并读取data进行预处理
        path = os.path.join(self.root, name)
        img = nib.load(path)
        data = img.get_fdata()
        data = data[None, ...]
        if self.transform is not None:
            data1 = self.transform(data.astype(np.float32()))

        #这里根据需要的列去依次读取数据
        #索引8是开始，15是数值的结尾，16是分类数据的开头
        tabular = torch.from_numpy(row[22:28].values.astype(np.float32))

        return data1, tabular,label,path
    def get_sample_weights(self, ratio={}):
        # ratio = {'PD':0.2} means averagely sample 2 PD cases and 8 no PD cases for each batch
        weights = []
        label_title = 'type'
        label_list = self.df[label_title].to_list()
        for i in range(len(label_list)):
            label_id = label_list[i]  # 获取当前标签
            if label_id == 'CN':
                label_list[i] = 0
            elif label_id == 'SMC':
                label_list[i] = 1
            elif label_id in ['MCI', 'LMCI', 'EMCI']:
                label_list[i] = 2
            else:
                print(f'Wrong label: {label_id}')  # 输出不符合预期的标签
                label_list[i] = None


        dataset_name_list = self.df['image id'].to_list()
        # if self.task in ratio:
        #     for i in label_list:
        #         if i == 0:
        #             weights.append(1 - ratio[self.task])
        #         elif i == 1:
        #             weights.append(ratio[self.task])
        #     return weights
        # if task is not in ratio, no specific value for the ratio, thus auto-balance the data according to data distribution
        unique = list(set(label_list))
        count = [float(label_list.count(a)) for a in unique]
        total = float(len(label_list))
        for i, name in zip(label_list, dataset_name_list):
            #if 'ADNI' in name: name = 'ADNI'
            # factor = self.task_config['sampleWeights'][name]
            factor = 1
            unique_idx = unique.index(i)
            weights.append(total / count[unique_idx] * factor)
        return weights

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n={len(self)}, "
                f"ptid_col='{self.ptid_col}', mri_col='{self.mri_col}', "
                f"return_dict={self.return_dict})")




class PTIDMRIDataset_paint(torch.utils.data.Dataset):
    def __init__(
            self,
            root:str,
            path: str,
            phase: str,
            ptid_col: str = "ptid",
            mri_col: str = "image id",
            label_col: str="type",
            deduplicate: bool = False,
            keep: Union[str, bool] = "first",
            return_dict: bool = False,
            transform: Optional[Callable[[Union[Tuple[str, str], Dict[str, str]]],
            Union[Tuple, Dict]]] = None,
    ):
        csv_path=os.path.join(path,phase)
        df = pd.read_csv(csv_path)

        # --- 列名自动识别（大小写不敏感 + 常见别名/模糊匹配） ---
        cols = list(df.columns)

        # 可选：按 PTID 去重（保留首次出现，保持原始顺序）
        if deduplicate:
            df = df.drop_duplicates(subset=[ptid_col], keep=keep)

        self.path=root
        self.df = df.reset_index(drop=True)
        self.ptid_col = ptid_col
        self.mri_col = mri_col
        self.root = root
        self.label_col = label_col
        self.return_dict = return_dict
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Union[Tuple[str, str], Dict[str, str]]:

        row = self.df.iloc[idx]
        # 转为字符串并处理缺失
        ptid = None if pd.isna(row[self.ptid_col]) else str(row[self.ptid_col])
        mri_id = None if pd.isna(row[self.mri_col]) else str(row[self.mri_col]).replace(",", "")

        label_id = None if pd.isna(row[self.label_col]) else str(row[self.label_col])
        #/data/birth/cyf/shared_data/CAPL/normalized_to_mni152/002_S_0295_I238627_MRI_brain.nii.gz
        if label_id =='CN':
            label=0
        elif label_id =='SMC':
            label=1
        elif label_id =='MCI' or label_id=='LMCI' or label_id=='EMCI':
            label=2
        else:
            print('wrong')
        name = f"{ptid}_{mri_id}_MRI_brain.nii.gz"
        #这里是根据id得到name，并读取data进行预处理
        path = os.path.join(self.root, name)
        img = nib.load(path)
        data = img.get_fdata()
        data = data[None, ...]
        if self.transform is not None:
            data1 = self.transform(data.astype(np.float32()))

        #这里根据需要的列去依次读取数据
        #索引8是开始，15是数值的结尾，16是分类数据的开头
        tabular = torch.from_numpy(row[22:28].values.astype(np.float32))

        return data1, tabular,label,ptid,mri_id
    def get_sample_weights(self, ratio={}):
        # ratio = {'PD':0.2} means averagely sample 2 PD cases and 8 no PD cases for each batch
        weights = []
        label_title = 'type'
        label_list = self.df[label_title].to_list()
        for i in range(len(label_list)):
            label_id = label_list[i]  # 获取当前标签
            if label_id == 'CN':
                label_list[i] = 0
            elif label_id == 'SMC':
                label_list[i] = 1
            elif label_id in ['MCI', 'LMCI', 'EMCI']:
                label_list[i] = 2
            else:
                print(f'Wrong label: {label_id}')  # 输出不符合预期的标签
                label_list[i] = None


        dataset_name_list = self.df['image id'].to_list()
        # if self.task in ratio:
        #     for i in label_list:
        #         if i == 0:
        #             weights.append(1 - ratio[self.task])
        #         elif i == 1:
        #             weights.append(ratio[self.task])
        #     return weights
        # if task is not in ratio, no specific value for the ratio, thus auto-balance the data according to data distribution
        unique = list(set(label_list))
        count = [float(label_list.count(a)) for a in unique]
        total = float(len(label_list))
        for i, name in zip(label_list, dataset_name_list):
            #if 'ADNI' in name: name = 'ADNI'
            # factor = self.task_config['sampleWeights'][name]
            factor = 1
            unique_idx = unique.index(i)
            weights.append(total / count[unique_idx] * factor)
        return weights

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(n={len(self)}, "
                f"ptid_col='{self.ptid_col}', mri_col='{self.mri_col}', "
                f"return_dict={self.return_dict})")