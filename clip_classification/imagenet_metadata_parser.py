"""
ImageNet元数据解析工具
用于从ILSVRC2012 devkit中提取类别名称和WNID的对应关系
"""

import os
import scipy.io
import numpy as np
from typing import Dict, List, Tuple


def parse_imagenet_metadata(meta_file: str) -> Dict[str, Dict]:
    """
    解析ImageNet的meta.mat文件
    
    Args:
        meta_file: meta.mat文件路径
    
    Returns:
        字典，key为WNID，value包含类别信息
    """
    # 加载.mat文件
    meta = scipy.io.loadmat(meta_file, squeeze_me=True, struct_as_record=False)
    
    # 提取synsets信息
    synsets = meta['synsets']
    
    # 构建WNID到类别信息的映射
    wnid_to_info = {}
    
    for synset in synsets:
        wnid = synset.WNID
        ilsvrc2012_id = synset.ILSVRC2012_ID
        words = synset.words
        gloss = synset.gloss if hasattr(synset, 'gloss') else ""
        num_children = synset.num_children
        
        wnid_to_info[wnid] = {
            'wnid': wnid,
            'ilsvrc2012_id': ilsvrc2012_id,
            'words': words,  # 类别名称（可能包含多个同义词，用逗号分隔）
            'gloss': gloss,  # 详细描述
            'num_children': num_children
        }
    
    return wnid_to_info


def get_class_name(wnid: str, metadata: Dict[str, Dict]) -> str:
    """
    根据WNID获取类别名称
    
    Args:
        wnid: WordNet ID (如 n01530575)
        metadata: 元数据字典
    
    Returns:
        类别名称字符串
    """
    if wnid in metadata:
        return metadata[wnid]['words']
    else:
        return f"Unknown ({wnid})"


def scan_imagenet_classes(train_dir: str) -> List[str]:
    """
    扫描ImageNet训练目录，获取所有类别的WNID
    
    Args:
        train_dir: 训练数据目录
    
    Returns:
        WNID列表
    """
    wnids = []
    for item in sorted(os.listdir(train_dir)):
        item_path = os.path.join(train_dir, item)
        if os.path.isdir(item_path) and item.startswith('n'):
            wnids.append(item)
    return wnids


def create_wnid_to_name_mapping(meta_file: str, train_dir: str) -> Dict[str, str]:
    """
    创建WNID到类别名称的映射
    
    Args:
        meta_file: meta.mat文件路径
        train_dir: 训练数据目录
    
    Returns:
        WNID到类别名称的字典
    """
    # 解析元数据
    print("解析ImageNet元数据...")
    metadata = parse_imagenet_metadata(meta_file)
    print(f"✓ 共加载 {len(metadata)} 个类别的元数据")
    
    # 扫描训练目录
    print("\n扫描训练数据目录...")
    wnids_in_data = scan_imagenet_classes(train_dir)
    print(f"✓ 发现 {len(wnids_in_data)} 个类别文件夹")
    
    # 创建映射
    wnid_to_name = {}
    missing_wnids = []
    
    for wnid in wnids_in_data:
        if wnid in metadata:
            wnid_to_name[wnid] = metadata[wnid]['words']
        else:
            wnid_to_name[wnid] = f"Unknown ({wnid})"
            missing_wnids.append(wnid)
    
    if missing_wnids:
        print(f"\n⚠ 警告: {len(missing_wnids)} 个WNID未在元数据中找到:")
        for wnid in missing_wnids[:5]:
            print(f"  - {wnid}")
        if len(missing_wnids) > 5:
            print(f"  ... 还有 {len(missing_wnids) - 5} 个")
    
    return wnid_to_name


def display_sample_mappings(wnid_to_name: Dict[str, str], num_samples: int = 20):
    """
    显示示例映射
    
    Args:
        wnid_to_name: WNID到名称的映射
        num_samples: 显示的样本数量
    """
    print("\n" + "="*80)
    print("WNID到类别名称映射示例:")
    print("="*80)
    
    for i, (wnid, name) in enumerate(sorted(wnid_to_name.items())[:num_samples]):
        print(f"{i+1:3d}. {wnid} => {name}")
    
    if len(wnid_to_name) > num_samples:
        print(f"... 还有 {len(wnid_to_name) - num_samples} 个类别")
    
    print("="*80)


def get_image_class_name(image_path: str, wnid_to_name: Dict[str, str]) -> str:
    """
    根据图像路径获取类别名称
    
    Args:
        image_path: 图像文件路径 (如 train/n01530575/n01530575_10007.JPEG)
        wnid_to_name: WNID到名称的映射
    
    Returns:
        类别名称
    """
    # 从路径中提取WNID
    parts = image_path.replace('\\', '/').split('/')
    
    # 查找包含WNID的部分（通常是倒数第二个）
    for part in reversed(parts):
        if part.startswith('n') and len(part) == 9:  # WNID格式: n########
            wnid = part
            return wnid_to_name.get(wnid, f"Unknown ({wnid})")
    
    # 也可以从文件名中提取
    filename = os.path.basename(image_path)
    if filename.startswith('n') and '_' in filename:
        wnid = filename.split('_')[0]
        return wnid_to_name.get(wnid, f"Unknown ({wnid})")
    
    return "Unknown"


def main():
    """主函数 - 演示如何使用"""
    
    # 路径配置
    META_FILE = r"G:\Thomas\3_1_project\data\ImageNet-data\meta\ILSVRC2012_devkit_t12\data\meta.mat"
    TRAIN_DIR = r"G:\Thomas\3_1_project\data\ImageNet-data\train"
    
    print("="*80)
    print("ImageNet 元数据解析工具")
    print("="*80)
    
    # 检查文件是否存在
    if not os.path.exists(META_FILE):
        print(f"❌ 错误: 找不到元数据文件: {META_FILE}")
        return
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ 错误: 找不到训练数据目录: {TRAIN_DIR}")
        return
    
    # 创建映射
    wnid_to_name = create_wnid_to_name_mapping(META_FILE, TRAIN_DIR)
    
    # 显示示例
    display_sample_mappings(wnid_to_name, num_samples=27)
    
    # 测试图像路径到类别名称的转换
    print("\n" + "="*80)
    print("图像路径到类别名称转换示例:")
    print("="*80)
    
    # 获取一些示例图像路径
    sample_wnids = list(wnid_to_name.keys())[:5]
    
    for wnid in sample_wnids:
        class_dir = os.path.join(TRAIN_DIR, wnid)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith('.JPEG')][:2]
            for img in images:
                img_path = os.path.join(class_dir, img)
                class_name = get_image_class_name(img_path, wnid_to_name)
                # 相对路径显示
                rel_path = os.path.relpath(img_path, TRAIN_DIR)
                print(f"  {rel_path}")
                print(f"    -> {class_name}\n")
    
    print("="*80)
    print("✓ 完成！")
    print("="*80)
    
    return wnid_to_name


if __name__ == '__main__':
    main()
