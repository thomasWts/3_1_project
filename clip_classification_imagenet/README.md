# CLIP + ImageNet 图像分类

基于PyTorch Dataset和CLIP的ImageNet数据加载器。

## 特性

✅ **标准PyTorch接口** - 继承自 `torch.utils.data.Dataset`  
✅ **自动元数据解析** - 从 `meta.mat` 自动加载WNID和类别名称  
✅ **灵活类别选择** - 支持按数量或WNID列表选择类别  
✅ **智能数据分割** - 自动处理有/无独立val目录  
✅ **CLIP预处理集成** - 内置CLIP归一化和数据增强  
✅ **完全可重复** - 支持随机种子设置  

## 快速开始

### 1. 基本用法

```python
from imagenet_dataset import create_imagenet_dataloaders

# 创建数据加载器
train_loader, val_loader = create_imagenet_dataloaders(
    root='G:/Thomas/3_1_project/data/ImageNet-data',
    batch_size=32,
    num_classes=20,      # 使用前20个类别
    use_clip_norm=True,  # 使用CLIP归一化
    seed=42              # 可重复性
)

# 使用数据
for images, labels in train_loader:
    # images: [B, 3, 224, 224]
    # labels: [B]
    pass
```

### 2. 指定特定类别

```python
# 通过WNID列表指定类别
selected_wnids = [
    "n01443537",  # goldfish
    "n01530575",  # brambling
    "n01729322",  # hognose snake
]

train_loader, val_loader = create_imagenet_dataloaders(
    root='path/to/imagenet',
    class_subset=selected_wnids,
    batch_size=16
)
```

### 3. 自定义Transform

```python
from torchvision import transforms

# 自定义数据增强
custom_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

train_loader, val_loader = create_imagenet_dataloaders(
    root='path/to/imagenet',
    transform_train=custom_transform,
    num_classes=50
)
```

### 4. 直接使用Dataset类

```python
from imagenet_dataset import ImageNetDataset

dataset = ImageNetDataset(
    root='path/to/imagenet',
    split='train',
    transform=custom_transform,
    num_classes=10
)

# 获取样本
image, label = dataset[0]

# 获取类别信息
class_name = dataset.get_class_name(label)
wnid = dataset.get_wnid(label)

# 获取详细信息
info = dataset.get_sample_info(0)
print(info['class_name'])
print(info['description'])
```

## 数据目录结构

```
ImageNet-data/
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
├── val/          # 可选，如果没有会自动从train分割
│   ├── n01440764/
│   └── ...
└── meta/
    └── ILSVRC2012_devkit_t12/
        └── data/
            └── meta.mat
```

## API参考

### create_imagenet_dataloaders()

创建训练和验证数据加载器。

**参数：**
- `root` (str): ImageNet数据根目录
- `batch_size` (int): 批次大小，默认32
- `num_workers` (int): 数据加载线程数，默认4
- `num_classes` (int, optional): 使用的类别数量（前N个）
- `class_subset` (List[str], optional): 指定的WNID列表
- `image_size` (int): 图像尺寸，默认224
- `use_clip_norm` (bool): 是否使用CLIP归一化，默认True
- `transform_train` (Compose, optional): 自定义训练变换
- `transform_val` (Compose, optional): 自定义验证变换
- `seed` (int, optional): 随机种子
- `val_split` (float): 如果没有val目录，从train分割的比例，默认0.1

**返回：**
- `train_loader`, `val_loader`: PyTorch DataLoader对象

### ImageNetDataset类

继承自 `torch.utils.data.Dataset`。

**方法：**
- `__len__()`: 返回数据集大小
- `__getitem__(idx)`: 获取样本 (image, label)
- `get_class_name(idx)`: 根据类别索引获取类别名称
- `get_wnid(idx)`: 根据类别索引获取WNID
- `get_sample_info(idx)`: 获取样本详细信息

## 运行示例

### 测试数据集类

```bash
python imagenet_dataset.py
```

### 运行演示Notebook

打开 `demo.ipynb` 查看完整示例，包括：
- 数据加载
- 可视化
- 类别统计
- 自定义用法

## 依赖

```
torch
torchvision
scipy
numpy
PIL
matplotlib (用于可视化)
```

## 注意事项

1. **Windows用户**：建议设置 `num_workers=0` 避免多进程问题
2. **CLIP归一化**：如果使用CLIP模型，请设置 `use_clip_norm=True`
3. **内存**：大批次或多类别会占用较多内存，根据GPU调整
4. **验证集**：如果没有独立val目录，会自动从train分割10%

## 测试结果

```
✓ 加载 train 集: 13000 张图像, 10 个类别
⚠ 未找到独立的val目录，从训练集中分割 10% 作为验证集
  训练集: 11700 样本
  验证集: 1300 样本

✓ 训练集批次数: 1463
✓ 验证集批次数: 163

测试数据加载...
  图像批次形状: torch.Size([8, 3, 224, 224])
  标签批次形状: torch.Size([8])
  图像数值范围: [-1.792, 2.132]
```

## 类别示例

| 索引 | WNID | 类别名称 |
|------|------|----------|
| 0 | n01443537 | goldfish, Carassius auratus |
| 1 | n01530575 | brambling, Fringilla montifringilla |
| 2 | n01687978 | agama |
| 3 | n01729322 | hognose snake, puff adder, sand viper |
| 4 | n01773797 | garden spider, Aranea diademata |

## License

MIT
