"""
使用CLIP模型在Tiny ImageNet-200上进行训练
支持从50类子集开始实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm
import os
from typing import Tuple, Optional
import argparse
import random
import numpy as np
from torchvision import transforms as T

from torch.cuda.amp import autocast, GradScaler

from tiny_imagenet_dataset import TinyImageNetDataset, create_dataloaders


class CLIPClassifier(nn.Module):
    """
    基于CLIP的图像分类器
    可选择性地冻结CLIP的部分参数
    """
    
    def __init__(
        self,
        num_classes: int,
        clip_model_name: str = "ViT-B/32",
        freeze_encoder: bool = False,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.device = device
        
        # 加载CLIP模型
        print(f"加载CLIP模型: {clip_model_name}...")
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        
        # 获取CLIP的特征维度
        if "ViT" in clip_model_name:
            feature_dim = 512
        else:  # RN50等
            feature_dim = 1024
        
        # 是否冻结CLIP编码器
        if freeze_encoder:
            print("冻结CLIP图像编码器参数")
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False
        else:
            print("微调CLIP图像编码器参数")
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes)
        )
        
        self.classifier.to(device)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: [batch_size, 3, H, W]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 使用CLIP提取图像特征
        image_features = self.clip_model.encode_image(images)
        
        # 归一化特征（CLIP的标准做法）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # 通过分类头
        logits = self.classifier(image_features.float())
        
        return logits


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: CLIPClassifier,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float]:
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert scaler is not None, "GradScaler is required when use_amp=True"
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: CLIPClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    desc: str = "Val"
) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(data_loader, desc=f"[{desc}]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_clip_classifier(
    data_root: str,
    num_classes: int = 50,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    freeze_encoder: bool = False,
    save_dir: str = './checkpoints',
    device: str = 'cuda',
    seed: int = 42,
    use_amp: bool = True,
    use_aug: bool = False,
):
    """
    训练CLIP分类器的主函数
    
    Args:
        data_root: 数据集根目录
        num_classes: 使用的类别数量
        batch_size: 批次大小
        num_epochs: 训练轮数
        lr: 学习率
        freeze_encoder: 是否冻结CLIP编码器
        save_dir: 模型保存目录
        device: 设备
    """
    
    print("="*70)
    print("CLIP + Tiny ImageNet-200 训练")
    print("="*70)
    print(f"类别数: {num_classes}")
    print(f"批次大小: {batch_size}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {lr}")
    print(f"冻结编码器: {freeze_encoder}")
    print(f"设备: {device}")
    print(f"随机种子: {seed}")
    print(f"AMP混合精度: {use_amp}")
    print(f"训练增强: {use_aug}")
    print("="*70)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 先创建模型以获取CLIP的预处理（包含Resize 224等）
    print("\n创建模型...")
    model = CLIPClassifier(
        num_classes=num_classes,
        clip_model_name="ViT-B/32",
        freeze_encoder=freeze_encoder,
        device=device
    )
    
    # 使用CLIP自带的preprocess作为基础，或构建增强版预处理
    clip_preprocess = model.preprocess
    if use_aug:
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        transform_train = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=clip_mean, std=clip_std),
        ])
        transform_eval = clip_preprocess
    else:
        transform_train = clip_preprocess
        transform_eval = clip_preprocess
    
    # 创建数据加载器（使用CLIP预处理，避免ViT位置编码尺寸不匹配）
    print("\n加载数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=0,  # Windows下设为0
        num_classes=num_classes,
        image_size=224,  # 不生效于clip_preprocess，仅作记录
        transform_train=transform_train,
        transform_eval=transform_eval,
        seed=seed,
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=use_amp)
    
    # 训练循环
    best_val_acc = 0.0
    best_epoch = 0
    
    print("\n开始训练...")
    print("="*70)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=use_amp, scaler=scaler
        )
        
        # 验证
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, desc="Val"
        )
        
        # 更新学习率
        scheduler.step()
        
        # 打印结果
        print(f"\nEpoch {epoch} 结果:")
        print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            
            save_path = os.path.join(save_dir, f'best_model_clip_{num_classes}class.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'num_classes': num_classes,
                'seed': seed,
                'use_amp': use_amp,
                'use_aug': use_aug,
            }, save_path)
            
            print(f"  ✓ 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
    
    # 训练完成
    print("\n" + "="*70)
    print("训练完成！")
    print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print("="*70)
    
    # 在测试集上评估（如果测试集有标签）
    # 注意: Tiny ImageNet的测试集没有公开标签，这里跳过
    
    return model, best_val_acc


def main():
    parser = argparse.ArgumentParser(description='训练CLIP on Tiny ImageNet-200')
    
    parser.add_argument('--data_root', type=str, 
                       default='g:/Thomas/3_1_project/data/tiny-imagenet-200',
                       help='数据集根目录')
    parser.add_argument('--num_classes', type=int, default=50,
                       help='使用的类别数量 (默认50，最大200)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='是否冻结CLIP编码器')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false', help='关闭AMP混合精度')
    parser.add_argument('--aug', dest='use_aug', action='store_true', help='启用训练时的数据增强（保持CLIP归一化）')
    parser.set_defaults(use_amp=True)
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    
    train_clip_classifier(
        data_root=args.data_root,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        save_dir=args.save_dir,
        device=device,
        seed=args.seed,
        use_amp=args.use_amp,
        use_aug=args.use_aug,
    )


if __name__ == '__main__':
    main()
