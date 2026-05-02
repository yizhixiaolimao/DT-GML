import os
import dicom2nifti


def convert_dcm_to_nii(dcm_dir, output_dir, compress=True):
    """
    将DICOM文件夹转换为NIfTI格式

    参数:
        dcm_dir (str): 包含DICOM文件的目录路径
        output_dir (str): 输出NIfTI文件的目录
        compress (bool): 是否压缩为.nii.gz格式(默认True)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 执行转换
    dicom2nifti.convert_directory(dcm_dir, output_dir, compression=compress)

    print(f"转换完成! 结果保存在: {output_dir}")


def find_deepest_dcm_folder(start_dir):
    """
    从start_dir开始查找包含DICOM文件的最深层文件夹

    参数:
        start_dir (str): 开始搜索的目录

    返回:
        str: 包含DICOM文件的最深层文件夹路径，如果没有则返回None
    """
    current_dcm_dir = None

    for dirpath, dirnames, filenames in os.walk(start_dir):
        # 检查当前目录是否包含DICOM文件
        dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]

        if dcm_files:
            current_dcm_dir = dirpath

    return current_dcm_dir


def process_adni_dataset(root_path):
    """
    处理ADNI数据集

    参数:
        root_path (str): ADNI数据的根目录路径
    """
    # 获取root_path下所有直接子文件夹
    subject_folders = [os.path.join(root_path, f) for f in os.listdir(root_path)
                       if os.path.isdir(os.path.join(root_path, f))]

    print(f"共找到 {len(subject_folders)} 个受试者文件夹")

    for subject_folder in subject_folders:
        print(f"\n处理受试者: {os.path.basename(subject_folder)}")

        # 查找该受试者文件夹下最深层的DICOM目录
        dcm_folder = find_deepest_dcm_folder(subject_folder)

        if dcm_folder:
            print(f"找到DICOM文件夹: {dcm_folder}")

            # 设置输出目录为受试者文件夹下的nii_output
            output_dir = os.path.join(subject_folder, "nii")

            try:
                # 执行转换
                convert_dcm_to_nii(dcm_folder, output_dir)

                # 打印转换结果
                converted_files = [f for f in os.listdir(output_dir)
                                   if f.endswith('.nii.gz')]
                print(f"生成的NIfTI文件: {converted_files}")

            except Exception as e:
                print(f"转换失败: {str(e)}")
        else:
            print("未找到DICOM文件")


if __name__ == "__main__":
    # ADNI数据根目录
    adni_root = "ADNI"

    # 检查路径是否存在
    if not os.path.exists(adni_root):
        print(f"错误: 路径 {adni_root} 不存在!")
    else:
        # 执行处理
        process_adni_dataset(adni_root)

