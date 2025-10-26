# Tiny ImageNet-200 + CLIP 训练项目

## 📁 已创建的文件

1. **tiny_imagenet_dataset.py** - 自定义Dataset类
2. **train_clip_imagenet.py** - CLIP训练脚本

## 🚀 快速开始

### 1. 测试Dataset类

```bash
# 在ML环境中运行
python tiny_imagenet_dataset.py
```

### 2. 开始训练（50类子集，推荐）

```bash
# 使用50个类别快速训练
python train_clip_imagenet.py --num_classes 50 --epochs 10 --batch_size 32
```

### 3. 训练完整数据集（200类）

```bash
# 使用全部200个类别训练
python train_clip_imagenet.py --num_classes 200 --epochs 20 --batch_size 32
```

### 4. 冻结CLIP编码器（更快训练）

```bash
# 只训练分类头，冻结CLIP特征提取器
python train_clip_imagenet.py --num_classes 50 --freeze_encoder --epochs 5
```

## 📊 数据集信息

- **数据集**: Tiny ImageNet-200
- **总类别数**: 200
- **图像尺寸**: 64×64
- **训练集**: 100,000 张图像 (每类500张)
- **验证集**: 10,000 张图像 (每类50张)
- **测试集**: 10,000 张图像 (无标签)

## 🎯 TinyImageNetDataset 类特性

### 主要功能

```python
from tiny_imagenet_dataset import TinyImageNetDataset, create_dataloaders

# 方式1: 直接使用Dataset
dataset = TinyImageNetDataset(
    root='g:/Thomas/3_1_project/data/tiny-imagenet-200',
    split='train',  # 'train', 'val', 'test'
    num_classes=50,  # 使用50个类别
    random_seed=42
)

# 方式2: 使用便捷函数创建DataLoader
train_loader, val_loader, test_loader = create_dataloaders(
    root='g:/Thomas/3_1_project/data/tiny-imagenet-200',
    batch_size=32,
    num_classes=50,
    image_size=64
)
```

### 支持的参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root` | str | - | 数据集根目录 |
| `split` | str | 'train' | 'train', 'val', 'test' |
| `transform` | Callable | None | 图像变换 |
| `num_classes` | int | 200 | 使用的类别数 (1-200) |
| `random_seed` | int | 42 | 随机种子 |

### 实用方法

```python
# 获取类别名称
class_name = dataset.get_class_name(label=0)  # "Egyptian cat"

# 获取WordNet ID
wnid = dataset.get_wnid(label=0)  # "n02124075"

# 获取样本
image, label = dataset[0]
```

## 🎓 CLIPClassifier 类特性

### 模型架构

```python
from train_clip_imagenet import CLIPClassifier

model = CLIPClassifier(
    num_classes=50,
    clip_model_name="ViT-B/32",  # 或 "RN50", "ViT-L/14"
    freeze_encoder=False,  # 是否冻结CLIP编码器
    device="cuda"
)
```

### 训练策略

1. **完全微调** (freeze_encoder=False)
   - 更新CLIP的所有参数
   - 需要更长时间，但效果更好
   - 推荐用于最终训练

2. **特征提取** (freeze_encoder=True)
   - 只训练分类头
   - 训练速度快，防止过拟合
   - 推荐用于快速实验

## 🔧 命令行参数

```bash
python train_clip_imagenet.py --help
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_root` | tiny-imagenet-200路径 | 数据集根目录 |
| `--num_classes` | 50 | 类别数量 |
| `--batch_size` | 32 | 批次大小 |
| `--epochs` | 10 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--freeze_encoder` | False | 是否冻结编码器 |
| `--save_dir` | ./checkpoints | 模型保存目录 |

## 📈 训练示例输出

```
======================================================================
CLIP + Tiny ImageNet-200 训练
======================================================================
类别数: 50
批次大小: 32
训练轮数: 10
学习率: 0.0001
冻结编码器: False
设备: cuda
======================================================================

加载数据集...
✓ 加载 train 集: 25000 张图像, 50 个类别
✓ 加载 val 集: 2500 张图像, 50 个类别
✓ 加载 test 集: 10000 张图像, 50 个类别

创建模型...
加载CLIP模型: ViT-B/32...
微调CLIP图像编码器参数

开始训练...
======================================================================

Epoch 1/10
----------------------------------------------------------------------
Epoch 1 [Train]: 100%|████████| 782/782 [02:15<00:00, loss=2.1234, acc=45.23%]
[Val]: 100%|████████| 79/79 [00:12<00:00, loss=1.8567, acc=52.34%]

Epoch 1 结果:
  训练 - Loss: 2.1234, Acc: 45.23%
  验证 - Loss: 1.8567, Acc: 52.34%
  学习率: 0.000100
  ✓ 保存最佳模型 (验证准确率: 52.34%)
...
```

## 💡 实验建议

### 快速实验（推荐先运行）
```bash
# 5分钟快速测试
python train_clip_imagenet.py \
    --num_classes 10 \
    --epochs 3 \
    --batch_size 64 \
    --freeze_encoder
```

### 中等规模实验
```bash
# 30-60分钟
python train_clip_imagenet.py \
    --num_classes 50 \
    --epochs 10 \
    --batch_size 32
```

### 完整实验
```bash
# 2-3小时
python train_clip_imagenet.py \
    --num_classes 200 \
    --epochs 20 \
    --batch_size 32 \
    --lr 5e-5
```

## 📝 保存的模型

训练完成后，模型保存在 `./checkpoints/` 目录：

```
checkpoints/
└── best_model_clip_50class.pth  # 50类最佳模型
```

加载模型示例：
```python
checkpoint = torch.load('checkpoints/best_model_clip_50class.pth')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"最佳验证准确率: {checkpoint['val_acc']:.2f}%")
```

## 🎨 自定义训练

可以通过修改代码来实现：
- 使用不同的CLIP模型 (RN50, ViT-L/14等)
- 调整数据增强策略
- 使用不同的优化器和学习率调度
- 添加更多的分类层

## ⚠️ 注意事项

1. **Windows系统**: 将 `num_workers` 设为 0
2. **内存不足**: 减小 `batch_size`
3. **训练太慢**: 使用 `--freeze_encoder` 或减少 `num_classes`
4. **显存溢出**: 减小 `batch_size` 或使用更小的CLIP模型

## 🔗 相关资源

- CLIP论文: https://arxiv.org/abs/2103.00020
- CLIP GitHub: https://github.com/openai/CLIP
- Tiny ImageNet: http://cs231n.stanford.edu/tiny-imagenet-200.zip

---

**祝训练顺利！** 🚀
