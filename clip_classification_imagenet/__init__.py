"""
CLIP + ImageNet 图像分类模块
"""

from .imagenet_dataset import ImageNetDataset, create_imagenet_dataloaders

__all__ = ['ImageNetDataset', 'create_imagenet_dataloaders']
