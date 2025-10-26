"""
使用CLIP在（部分）ImageNet数据上训练一个线性分类头
- 复用自定义 ImageNetDataset / DataLoader 工厂
- 支持可重复随机种子、AMP混合精度、CLIP对齐的数据增强
"""

import os
import time
import argparse
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import clip
from imagenet_dataset import create_imagenet_dataloaders


# ------------------------- 实用函数 -------------------------
def set_seed(seed: int) -> None:
    """设置随机种子，确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------- 模型定义 -------------------------
class CLIPClassifier(nn.Module):
    """基于CLIP视觉编码器 + 线性分类头"""

    def __init__(self, num_classes: int, clip_arch: str = "ViT-B/32", freeze_encoder: bool = True, device: str = "cuda"):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # 加载CLIP模型与预处理
        print(f"加载CLIP模型: {clip_arch} ...")
        self.clip_model, self.preprocess = clip.load(clip_arch, device=device)

        # 冻结或微调视觉编码器
        self.visual_frozen = freeze_encoder
        if freeze_encoder:
            print("冻结CLIP视觉编码器参数")
            # 冻结CLIP模型的所有参数（包含文本分支），避免无用参数进入优化器
            for p in self.clip_model.parameters():
                p.requires_grad = False
        else:
            print("微调CLIP视觉编码器参数")

        # 决定特征维度
        if "ViT" in clip_arch:
            feat_dim = 512
        else:
            feat_dim = 1024  # RN50 等

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(feat_dim, num_classes)
        )
        self.classifier.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 视觉编码器前向
        with torch.no_grad() if self.visual_frozen else torch.enable_grad():
            feats = self.clip_model.encode_image(images)
        feats = feats.float()  # 保险，确保fp32用于线性头
        logits = self.classifier(feats)
        return logits


# ------------------------- 训练/评估 -------------------------
from torch.amp import autocast, GradScaler  # torch>=2.0 推荐API

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    scaler: Optional[GradScaler] = None,
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="[Train]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            assert scaler is not None
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/total:.2f}%")

    return running_loss / len(dataloader), 100.0 * correct / max(1, total)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="[Val]")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        if use_amp:
            with autocast(device_type="cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0*correct/total:.2f}%")
    return running_loss / len(dataloader), 100.0 * correct / max(1, total)


# ------------------------- 主训练流程 -------------------------
def train(
    data_root: str,
    output_dir: str,
    device: torch.device,
    arch: str = "ViT-B/32",
    num_classes: Optional[int] = 1000,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    freeze_encoder: bool = True,
    num_workers: int = 0,
    class_subset: Optional[List[str]] = None,
    seed: int = 42,
    use_amp: bool = True,
    use_aug: bool = True,
    val_split: float = 0.1,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    # 先加载一次CLIP，拿到其预处理（224 + CLIP归一化）
    tmp_model, clip_preprocess = clip.load(arch, device=device)
    del tmp_model  # 仅为获得preprocess

    # 训练增强（CLIP对齐）
    if use_aug:
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        from torchvision import transforms as T
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

    # 创建数据加载器
    train_loader, val_loader = create_imagenet_dataloaders(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        num_classes=num_classes,
        class_subset=class_subset,
        image_size=224,
        use_clip_norm=True,
        transform_train=transform_train,
        transform_val=transform_eval,
        seed=seed,
        val_split=val_split,
    )

    # 模型
    model = CLIPClassifier(num_classes=(num_classes if num_classes is not None else 1000),
                           clip_arch=arch, freeze_encoder=freeze_encoder, device=device)

    # 优化器/调度器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=use_amp)

    # 训练循环
    best_val, best_path = 0.0, os.path.join(output_dir, f"best_clip_imagenet_{num_classes or 'all'}cls.pth")
    print("\n开始训练 ...")
    print("="*70)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, use_amp=use_amp, scaler=scaler)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        scheduler.step()
        dt = time.time() - t0
        print(f"  {dt:.1f}s | Train {train_loss:.4f}/{train_acc:.2f}% | Val {val_loss:.4f}/{val_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': best_val,
                'arch': arch,
                'num_classes': num_classes,
                'seed': seed,
                'use_amp': use_amp,
                'use_aug': use_aug,
                'freeze_encoder': freeze_encoder,
            }, best_path)
            print(f"  ✓ 保存最佳模型 -> {best_path}")
    print("="*70)
    print(f"训练完成，最佳验证准确率: {best_val:.2f}%")
    return best_path


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP classifier on (partial) ImageNet")
    p.add_argument('--data_root', type=str, default=os.path.join('G:', os.sep, 'Thomas', '3_1_project', 'data', 'ImageNet-data'))
    p.add_argument('--output_dir', type=str, default=os.path.join('.', 'checkpoints'))
    p.add_argument('--arch', type=str, default='ViT-B/32')
    p.add_argument('--num_classes', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--freeze_encoder', action='store_true')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--classes', type=str, nargs='*', default=None, help='可选的WNID列表，用于选择类别子集')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', dest='use_amp', action='store_false')
    p.add_argument('--aug', dest='use_aug', action='store_true')
    p.add_argument('--val_split', type=float, default=0.1, help='当没有val目录时，从train划分的比例')
    p.set_defaults(use_amp=True)
    return p.parse_args()


def main():
    args = parse_args()

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 种子
    set_seed(args.seed)

    # 打印配置
    print("="*70)
    print("CLIP + ImageNet 训练配置")
    print("="*70)
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录:   {args.output_dir}")
    print(f"架构:       {args.arch}")
    print(f"类别数:     {args.num_classes}")
    print(f"批次大小:   {args.batch_size}")
    print(f"训练轮数:   {args.epochs}")
    print(f"学习率:     {args.lr}")
    print(f"权重衰减:   {args.weight_decay}")
    print(f"冻结编码器: {args.freeze_encoder}")
    print(f"AMP:        {args.use_amp}")
    print(f"增强:       {args.use_aug}")
    print(f"VAL划分:    {args.val_split}")
    print(f"设备:       {device}")
    print("="*70)

    best = train(
        data_root=args.data_root,
        output_dir=args.output_dir,
        device=device,
        arch=args.arch,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        num_workers=args.num_workers,
        class_subset=args.classes,
        seed=args.seed,
        use_amp=args.use_amp,
        use_aug=args.use_aug,
        val_split=args.val_split,
    )
    print(f"最佳模型: {best}")


if __name__ == '__main__':
    main()
