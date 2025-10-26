"""
Tiny ImageNet-200 Dataset类
用于加载和处理Tiny ImageNet-200数据集
支持训练/验证/测试集，可与CLIP模型配合使用
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from typing import Optional, Callable, Tuple, List
import random


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet-200 数据集类
    
    数据集结构:
    tiny-imagenet-200/
        ├── train/
        │   ├── n01443537/
        │   │   └── images/
        │   │       └── *.JPEG
        ├── val/
        │   ├── images/
        │   │   └── *.JPEG
        │   └── val_annotations.txt
        └── test/
            └── images/
                └── *.JPEG
    
    Args:
        root (str): 数据集根目录路径
        split (str): 'train', 'val', 'test' 之一
        transform (Optional[Callable]): 图像变换函数
        num_classes (int): 使用的类别数量，默认200（全部），可以设置更小的值用于实验
        random_seed (int): 随机种子，用于选择类别子集
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        num_classes: int = 200,
        random_seed: int = 42
    ):
        super().__init__()
        
        assert split in ['train', 'val', 'test'], "split必须是'train', 'val', 'test'之一"
        assert num_classes <= 200, "num_classes不能超过200"
        
        self.root = root
        self.split = split
        self.transform = transform
        self.num_classes = num_classes
        
        # 读取类别信息
        self.wnids = self._load_wnids()
        
        # 如果需要使用子集，随机选择类别
        if num_classes < 200:
            random.seed(random_seed)
            self.wnids = random.sample(self.wnids, num_classes)
            self.wnids.sort()
        
        # 创建类别到索引的映射
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        
        # 读取类别名称
        self.class_names = self._load_class_names()
        
        # 加载数据
        self.samples = self._load_samples()
        
        print(f"✓ 加载 {split} 集: {len(self.samples)} 张图像, {len(self.wnids)} 个类别")
    
    def _load_wnids(self) -> List[str]:
        """加载类别ID列表"""
        wnids_file = os.path.join(self.root, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        return wnids
    
    def _load_class_names(self) -> dict:
        """加载类别名称映射"""
        words_file = os.path.join(self.root, 'words.txt')
        class_names = {}
        
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    wnid, name = parts
                    if wnid in self.class_to_idx:
                        class_names[wnid] = name
        
        return class_names
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """加载数据样本列表 (image_path, label)"""
        samples = []
        
        if self.split == 'train':
            # 训练集：每个类别一个文件夹
            train_dir = os.path.join(self.root, 'train')
            for wnid in self.wnids:
                class_dir = os.path.join(train_dir, wnid, 'images')
                if not os.path.exists(class_dir):
                    continue
                
                label = self.class_to_idx[wnid]
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.JPEG'):
                        img_path = os.path.join(class_dir, img_name)
                        samples.append((img_path, label))
        
        elif self.split == 'val':
            # 验证集：需要读取标注文件
            val_dir = os.path.join(self.root, 'val', 'images')
            anno_file = os.path.join(self.root, 'val', 'val_annotations.txt')
            
            with open(anno_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    img_name, wnid = parts[0], parts[1]
                    
                    # 只加载选中类别的样本
                    if wnid in self.class_to_idx:
                        img_path = os.path.join(val_dir, img_name)
                        label = self.class_to_idx[wnid]
                        samples.append((img_path, label))
        
        else:  # test
            # 测试集：没有标签
            test_dir = os.path.join(self.root, 'test', 'images')
            for img_name in os.listdir(test_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(test_dir, img_name)
                    samples.append((img_path, -1))  # 测试集无标签
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本
        
        Returns:
            image: 转换后的图像tensor
            label: 类别标签 (测试集返回-1)
        """
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, label: int) -> str:
        """根据标签获取类别名称"""
        wnid = self.wnids[label]
        return self.class_names.get(wnid, wnid)
    
    def get_wnid(self, label: int) -> str:
        """根据标签获取WordNet ID"""
        return self.wnids[label]


def get_transforms(image_size: int = 64, is_training: bool = True) -> transforms.Compose:
    """
    获取图像变换
    
    Args:
        image_size: 目标图像尺寸
        is_training: 是否为训练模式（训练模式会使用数据增强）
    
    Returns:
        transforms.Compose: 变换组合
    """
    if is_training:
        # 训练集：使用数据增强
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证/测试集：只做基本变换
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    num_classes: int = 200,
    image_size: int = 64,
    random_seed: int = 42,
    transform_train: Optional[transforms.Compose] = None,
    transform_eval: Optional[transforms.Compose] = None,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练/验证/测试集的DataLoader
    
    Args:
        root: 数据集根目录
        batch_size: 批次大小
        num_workers: 数据加载的工作进程数
        num_classes: 使用的类别数（默认200，可设置为50等进行实验）
        image_size: 图像尺寸
        random_seed: 随机种子
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # 创建数据集
    train_dataset = TinyImageNetDataset(
        root=root,
        split='train',
        transform=transform_train if transform_train is not None else get_transforms(image_size, is_training=True),
        num_classes=num_classes,
        random_seed=random_seed
    )
    
    val_dataset = TinyImageNetDataset(
        root=root,
        split='val',
        transform=transform_eval if transform_eval is not None else get_transforms(image_size, is_training=False),
        num_classes=num_classes,
        random_seed=random_seed
    )
    
    test_dataset = TinyImageNetDataset(
        root=root,
        split='test',
        transform=transform_eval if transform_eval is not None else get_transforms(image_size, is_training=False),
        num_classes=num_classes,
        random_seed=random_seed
    )
    
    # DataLoader 随机性控制
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init,
    )
    
    return train_loader, val_loader, test_loader


# 使用示例
if __name__ == '__main__':
    # 数据集路径
    data_root = 'g:/Thomas/3_1_project/data/tiny-imagenet-200'
    
    print("="*70)
    print("Tiny ImageNet-200 Dataset 测试")
    print("="*70)
    
    # 测试1: 加载完整数据集（200类）
    print("\n测试1: 加载完整数据集（200类）")
    train_loader, val_loader, test_loader = create_dataloaders(
        root=data_root,
        batch_size=32,
        num_workers=0,  # Windows下设置为0避免多进程问题
        num_classes=200,
        image_size=64
    )
    
    # 打印数据集信息
    print(f"\n训练集batch数: {len(train_loader)}")
    print(f"验证集batch数: {len(val_loader)}")
    print(f"测试集batch数: {len(test_loader)}")
    
    # 测试2: 加载一个batch
    print("\n测试2: 加载一个batch")
    images, labels = next(iter(train_loader))
    print(f"图像shape: {images.shape}")
    print(f"标签shape: {labels.shape}")
    print(f"标签范围: [{labels.min()}, {labels.max()}]")
    
    # 测试3: 查看类别名称
    print("\n测试3: 前10个类别名称")
    dataset = train_loader.dataset
    for i in range(min(10, dataset.num_classes)):
        wnid = dataset.get_wnid(i)
        class_name = dataset.get_class_name(i)
        print(f"  {i}: {wnid} - {class_name}")
    
    # 测试4: 加载50类子集（用于实验）
    print("\n测试4: 加载50类子集")
    train_loader_50, val_loader_50, test_loader_50 = create_dataloaders(
        root=data_root,
        batch_size=32,
        num_workers=0,
        num_classes=50,  # 只使用50个类别
        image_size=64
    )
    
    print(f"训练集batch数: {len(train_loader_50)}")
    print(f"验证集batch数: {len(val_loader_50)}")
    
    print("\n✅ 测试完成！")
