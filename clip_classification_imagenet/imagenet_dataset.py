"""
ImageNet数据集加载器
支持CLIP模型的ImageNet-1K数据集加载和预处理
"""

import os
import random
from typing import Tuple, Optional, Callable, List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io
import numpy as np


class ImageNetMetadata:
    """
    ImageNet元数据管理类
    负责解析meta.mat文件，提供WNID到类别名称的映射
    """
    
    def __init__(self, meta_file: str):
        """
        初始化元数据
        
        Args:
            meta_file: meta.mat文件路径
        """
        self.meta_file = meta_file
        self.wnid_to_info = self._parse_metadata()
        self.wnid_to_idx = {}  # WNID到索引的映射
        self.idx_to_wnid = {}  # 索引到WNID的映射
    
    def _parse_metadata(self) -> Dict[str, Dict]:
        """解析meta.mat文件"""
        if not os.path.exists(self.meta_file):
            raise FileNotFoundError(f"元数据文件不存在: {self.meta_file}")
        
        meta = scipy.io.loadmat(self.meta_file, squeeze_me=True, struct_as_record=False)
        synsets = meta['synsets']
        
        wnid_to_info = {}
        for synset in synsets:
            wnid = synset.WNID
            wnid_to_info[wnid] = {
                'wnid': wnid,
                'ilsvrc2012_id': synset.ILSVRC2012_ID,
                'words': synset.words,
                'gloss': synset.gloss if hasattr(synset, 'gloss') else "",
            }
        
        return wnid_to_info
    
    def build_class_mapping(self, wnids: List[str]) -> None:
        """
        构建类别索引映射
        
        Args:
            wnids: WNID列表（按字母顺序排列）
        """
        self.wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.idx_to_wnid = {idx: wnid for idx, wnid in enumerate(wnids)}
    
    def get_class_name(self, wnid: str) -> str:
        """获取类别名称"""
        if wnid in self.wnid_to_info:
            return self.wnid_to_info[wnid]['words']
        return f"Unknown ({wnid})"
    
    def get_class_description(self, wnid: str) -> str:
        """获取类别描述"""
        if wnid in self.wnid_to_info:
            return self.wnid_to_info[wnid]['gloss']
        return ""
    
    def get_idx_from_wnid(self, wnid: str) -> int:
        """从WNID获取类别索引"""
        return self.wnid_to_idx.get(wnid, -1)
    
    def get_wnid_from_idx(self, idx: int) -> str:
        """从类别索引获取WNID"""
        return self.idx_to_wnid.get(idx, "")


class ImageNetDataset(Dataset):
    """
    ImageNet数据集类
    
    支持:
    - 自动解析WNID和类别名称
    - CLIP预处理集成
    - 灵活的数据增强
    - 类别子集选择
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        meta_file: Optional[str] = None,
        num_classes: Optional[int] = None,
        class_subset: Optional[List[str]] = None,
    ):
        """
        初始化ImageNet数据集
        
        Args:
            root: ImageNet数据根目录
            split: 数据集划分 ('train', 'val')
            transform: 图像变换函数
            meta_file: meta.mat文件路径（如果为None，自动查找）
            num_classes: 使用的类别数量（取前N个类别）
            class_subset: 指定的类别WNID列表
        
        数据目录结构:
            root/
            ├── train/
            │   ├── n01440764/
            │   │   ├── n01440764_10026.JPEG
            │   │   └── ...
            │   ├── n01443537/
            │   └── ...
            ├── val/
            │   ├── n01440764/
            │   └── ...
            └── meta/
                └── ILSVRC2012_devkit_t12/
                    └── data/
                        └── meta.mat
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # 构建数据目录路径
        self.data_dir = os.path.join(root, split)
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 自动查找meta.mat文件
        if meta_file is None:
            meta_file = os.path.join(
                root, 
                "meta", 
                "ILSVRC2012_devkit_t12", 
                "data", 
                "meta.mat"
            )
        
        # 加载元数据
        self.metadata = ImageNetMetadata(meta_file)
        
        # 扫描类别文件夹
        all_wnids = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('n')
        ])
        
        # 选择类别子集
        if class_subset is not None:
            self.wnids = [w for w in class_subset if w in all_wnids]
        elif num_classes is not None:
            self.wnids = all_wnids[:num_classes]
        else:
            self.wnids = all_wnids
        
        # 构建类别索引映射
        self.metadata.build_class_mapping(self.wnids)
        self.num_classes = len(self.wnids)
        
        # 加载所有图像路径和标签
        self.samples = []
        self._load_samples()
        
        print(f"✓ 加载 {split} 集: {len(self.samples)} 张图像, {self.num_classes} 个类别")
    
    def _load_samples(self):
        """扫描并加载所有图像路径和标签"""
        for wnid in self.wnids:
            class_dir = os.path.join(self.data_dir, wnid)
            class_idx = self.metadata.get_idx_from_wnid(wnid)
            
            # 获取该类别的所有图像
            image_files = [
                f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpeg', '.jpg', '.png'))
            ]
            
            # 添加到样本列表
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                self.samples.append((img_path, class_idx, wnid))
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
        
        Returns:
            (image, label): 图像张量和类别索引
        """
        img_path, label, wnid = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"⚠ 无法加载图像 {img_path}: {e}")
            # 返回一个黑色图像作为占位
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx: int) -> str:
        """
        根据类别索引获取类别名称
        
        Args:
            idx: 类别索引
        
        Returns:
            类别名称
        """
        wnid = self.metadata.get_wnid_from_idx(idx)
        return self.metadata.get_class_name(wnid)
    
    def get_wnid(self, idx: int) -> str:
        """
        根据类别索引获取WNID
        
        Args:
            idx: 类别索引
        
        Returns:
            WNID
        """
        return self.metadata.get_wnid_from_idx(idx)
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        获取样本的详细信息
        
        Args:
            idx: 样本索引
        
        Returns:
            包含路径、标签、WNID、类别名称的字典
        """
        img_path, label, wnid = self.samples[idx]
        return {
            'path': img_path,
            'label': label,
            'wnid': wnid,
            'class_name': self.metadata.get_class_name(wnid),
            'description': self.metadata.get_class_description(wnid)
        }


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_clip_norm: bool = True
) -> transforms.Compose:
    """
    获取图像变换
    
    Args:
        image_size: 目标图像尺寸
        is_training: 是否为训练模式
        use_clip_norm: 是否使用CLIP的归一化参数
    
    Returns:
        transforms.Compose对象
    """
    # CLIP的归一化参数
    if use_clip_norm:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    else:
        # ImageNet标准归一化
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    if is_training:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform


def create_imagenet_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: Optional[int] = None,
    class_subset: Optional[List[str]] = None,
    image_size: int = 224,
    use_clip_norm: bool = True,
    transform_train: Optional[transforms.Compose] = None,
    transform_val: Optional[transforms.Compose] = None,
    seed: Optional[int] = None,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建ImageNet数据加载器
    
    Args:
        root: ImageNet数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        num_classes: 使用的类别数量
        class_subset: 指定的类别WNID列表
        image_size: 图像尺寸
        use_clip_norm: 是否使用CLIP归一化
        transform_train: 自定义训练集变换（如果为None，使用默认）
        transform_val: 自定义验证集变换（如果为None，使用默认）
        seed: 随机种子
        val_split: 如果没有独立val目录，从train中分割的比例
    
    Returns:
        (train_loader, val_loader)
    """
    # 使用自定义变换或默认变换
    if transform_train is None:
        transform_train = get_transforms(image_size, is_training=True, use_clip_norm=use_clip_norm)
    
    if transform_val is None:
        transform_val = get_transforms(image_size, is_training=False, use_clip_norm=use_clip_norm)
    
    # 检查是否有独立的val目录
    val_dir = os.path.join(root, 'val')
    has_val_dir = os.path.exists(val_dir) and os.path.isdir(val_dir)
    
    # 创建训练数据集
    train_dataset = ImageNetDataset(
        root=root,
        split='train',
        transform=transform_train,
        num_classes=num_classes,
        class_subset=class_subset,
    )
    
    # 创建验证数据集
    if has_val_dir:
        # 使用独立的val目录
        val_dataset = ImageNetDataset(
            root=root,
            split='val',
            transform=transform_val,
            num_classes=num_classes,
            class_subset=class_subset,
        )
    else:
        # 从train中分割验证集
        print(f"⚠ 未找到独立的val目录，从训练集中分割 {val_split*100:.0f}% 作为验证集")
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        if seed is not None:
            torch.manual_seed(seed)
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, 
            [train_size, val_size]
        )
        
        # 为验证集设置不同的transform
        # 注意：这里需要重新包装dataset以应用不同的transform
        print(f"  训练集: {train_size} 样本")
        print(f"  验证集: {val_size} 样本")
    
    # 随机数生成器（用于可重复性）
    generator = None
    worker_init = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        def _seed_worker(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed % (2**32 - 1))
            random.seed(worker_seed)
        
        worker_init = _seed_worker
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    """测试代码"""
    
    print("="*70)
    print("ImageNet数据集加载器测试")
    print("="*70)
    
    # 配置
    DATA_ROOT = r"G:\Thomas\3_1_project\data\ImageNet-data"
    BATCH_SIZE = 8
    NUM_CLASSES = 10  # 使用前10个类别测试
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader, val_loader = create_imagenet_dataloaders(
        root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        num_workers=0,
        num_classes=NUM_CLASSES,
        image_size=224,
        use_clip_norm=True,
        seed=42,
    )
    
    print(f"\n✓ 训练集批次数: {len(train_loader)}")
    print(f"✓ 验证集批次数: {len(val_loader)}")
    
    # 获取一个批次测试
    print("\n测试数据加载...")
    images, labels = next(iter(train_loader))
    print(f"  图像批次形状: {images.shape}")
    print(f"  标签批次形状: {labels.shape}")
    print(f"  图像数值范围: [{images.min():.3f}, {images.max():.3f}]")
    
    # 显示类别信息
    print("\n类别信息:")
    print("="*70)
    # 获取原始dataset（处理可能的Subset包装）
    base_dataset = train_loader.dataset
    if hasattr(base_dataset, 'dataset'):
        base_dataset = base_dataset.dataset
    
    for i in range(min(10, base_dataset.num_classes)):
        wnid = base_dataset.get_wnid(i)
        name = base_dataset.get_class_name(i)
        print(f"  [{i:2d}] {wnid}: {name}")
    
    # 显示样本详细信息
    print("\n样本详细信息（前3个）:")
    print("="*70)
    for i in range(3):
        info = base_dataset.get_sample_info(i)
        print(f"样本 {i}:")
        print(f"  路径: {os.path.basename(info['path'])}")
        print(f"  WNID: {info['wnid']}")
        print(f"  标签: {info['label']}")
        print(f"  类别: {info['class_name']}")
        print(f"  描述: {info['description']}")
        print()
    
    print("="*70)
    print("✓ 测试完成！")
    print("="*70)
