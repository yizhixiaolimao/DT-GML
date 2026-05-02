import os
import random
import shutil

# 定义路径
cod = 'scn'
pet_source_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/pet/", cod)
mri_source_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/mri/", cod)
pet_test_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/pet/test/", cod)
pet_train_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/pet/train/", cod)
mri_test_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/mri/test/", cod)
mri_train_dir = os.path.join("/Data/Users/cyf/shared_data/CAPL/new_register/mri/train/", cod)

# 创建目标目录（如果不存在）
os.makedirs(pet_test_dir, exist_ok=True)
os.makedirs(pet_train_dir, exist_ok=True)
os.makedirs(mri_test_dir, exist_ok=True)
os.makedirs(mri_train_dir, exist_ok=True)

# 获取所有 PET 的 .nii.gz 文件
pet_files = [f for f in os.listdir(pet_source_dir) if f.endswith('.nii.gz')]
pet_file_paths = [os.path.join(pet_source_dir, f) for f in pet_files]



# 随机选择 9 个 PET 文件（用于测试集）
random.seed(42)  # 设置随机种子以确保可重复性
test_pet_files = random.sample(pet_file_paths, 44)
test_pet_names = [os.path.basename(f) for f in test_pet_files]  # 提取文件名

# 移动 PET 和 MRI 文件
for pet_path in pet_file_paths:
    pet_filename = os.path.basename(pet_path)
    mri_path = os.path.join(mri_source_dir, pet_filename)  # 对应的 MRI 文件路径

    # 检查 MRI 文件是否存在
    if not os.path.exists(mri_path):
        print(f"警告：MRI 文件 {pet_filename} 不存在，跳过")
        continue

    # 移动 PET 文件
    if pet_path in test_pet_files:
        shutil.move(pet_path, os.path.join(pet_test_dir, pet_filename))
        shutil.move(mri_path, os.path.join(mri_test_dir, pet_filename))  # 移动对应的 MRI 文件
    else:
        shutil.move(pet_path, os.path.join(pet_train_dir, pet_filename))
        shutil.move(mri_path, os.path.join(mri_train_dir, pet_filename))  # 移动对应的 MRI 文件

print("操作完成：")
print(f"移动到 PET test 目录: {len(test_pet_files)} 个文件")
print(f"移动到 PET train 目录: {len(pet_file_paths) - len(test_pet_files)} 个文件")
print(f"同时移动了对应的 MRI 文件")