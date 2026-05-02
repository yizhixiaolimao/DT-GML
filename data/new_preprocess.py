import os
import glob
import ants


def register_mri_to_pet(mri_path, pet_path, mri_output_path, pet_output_path=None):
    # 读取图像
    pet = ants.image_read(pet_path)
    mri = ants.image_read(mri_path)

    # 执行配准
    reg_result = ants.registration(
        fixed=pet,
        moving=mri,
        type_of_transform='SyN',
        verbose=True
    )

    # 保存配准后的MRI
    ants.image_write(reg_result['warpedmovout'], mri_output_path)

    # 同时保存PET到新路径
    if pet_output_path:
        os.makedirs(os.path.dirname(pet_output_path), exist_ok=True)
        ants.image_write(pet, pet_output_path)
def mri_find_nii_files_glob(folder):
    """使用glob查找所有.nii文件"""
    return glob.glob(os.path.join(folder, '**/*.nii'), recursive=True)
def pet_find_nii_files_glob(folder):
    """使用glob查找所有.nii文件"""
    return glob.glob(os.path.join(folder, '**/*.nii.gz'), recursive=True)

ori_path = "/Data/Users/cyf/shared_data/CAPL/normal_pet/"
#什么状态
cod='scn'

path=os.path.join(ori_path, cod,'ADNI')
mri_out=os.path.join('/Data/Users/cyf/shared_data/CAPL/new_register/mri',cod)
pet_out=os.path.join('/Data/Users/cyf/shared_data/CAPL/new_register/pet',cod)


# 获取所有子文件夹的完整路径
subfolders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print(len(subfolders))

#print(len(subfolders))
mri_path = []
pet_path = []
for subfolder in subfolders:
    mri_subfolder = subfolder.replace("/normal_pet/", "/normal_mri/")
    mri_folder = os.path.join(mri_subfolder,'nii')
    pet_folder = os.path.join(subfolder,'reg')
    mri_files_glob = mri_find_nii_files_glob(mri_folder)[0]
    pet_files_glob = pet_find_nii_files_glob(pet_folder)[0]
    mri_path.append(mri_files_glob)
    pet_path.append(pet_files_glob)
print('mri:',len(mri_path))
print('pet:',len(pet_path))
for i in range(len(mri_path)):
    input_mri = mri_path[i]
    input_pet = pet_path[i]
    name = str(i) + '.nii.gz'
    output_path_mri = os.path.join(mri_out, name)
    output_path_pet = os.path.join(pet_out, name)
    register_mri_to_pet(input_mri, input_pet, output_path_mri, output_path_pet)