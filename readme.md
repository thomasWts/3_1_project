# Pokemon图像分类深度学习项目

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 项目简介

本项目是一个基于深度学习的Pokemon图像分类系统，使用PyTorch框架实现。项目包含了从头训练和迁移学习两种方法，主要使用ResNet18网络结构对Pokemon图像进行5分类识别。

支持的Pokemon类别：
- Bulbasaur (妙蛙种子) 🌱
- Charmander (小火龙) 🔥  
- Mewtwo (超梦) 🧠
- Pikachu (皮卡丘) ⚡
- Squirtle (杰尼龟) 💧

## 数据集展示

本项目使用的Pokemon数据集包含5个不同的Pokemon类别，每个类别包含多张不同角度和风格的图像：

### 数据集样本展示
以下是各个类别的代表性图像：

| Pokemon类别 | 样本图像 | 描述 |
|-------------|----------|------|
| Bulbasaur (妙蛙种子) | ![Bulbasaur](data/pokemon/bulbasaur/00000000.png) | 草系宝可梦，背上有球根 |
| Charmander (小火龙) | ![Charmander](data/pokemon/charmander/00000000.png) | 火系宝可梦，尾巴有火焰 |
| Mewtwo (超梦) | ![Mewtwo](data/pokemon/mewtwo/00000000.jpg) | 超能力系传说宝可梦 |
| Pikachu (皮卡丘) | ![Pikachu](data/pokemon/pikachu/00000000.png) | 电系宝可梦，招牌角色 |
| Squirtle (杰尼龟) | ![Squirtle](data/pokemon/squirtle/00000000.png) | 水系宝可梦，有龟壳 |

### 数据集统计
- **总图像数量**: 约1167张图像
- **图像格式**: PNG, JPG, JPEG
- **数据划分**: 训练集60% / 验证集20% / 测试集20%
- **图像尺寸**: 统一resize到224×224像素

## 项目结构

```
3_1_project/
├── best_resnet18.pth          # 训练好的最佳模型权重
├── readme.md                  # 项目说明文档
├── data/                      # 数据集目录
│   ├── FashionMNIST/         # FashionMNIST数据集
│   └── pokemon/              # Pokemon图像数据集
│       ├── images.csv        # 图像索引文件
│       ├── bulbasaur/        # 妙蛙种子图像
│       ├── charmander/       # 小火龙图像
│       ├── mewtwo/           # 超梦图像
│       ├── pikachu/          # 皮卡丘图像
│       └── squirtle/         # 杰尼龟图像
├── image_classification/      # 主要代码目录
│   ├── __init__.py
│   ├── data.py               # 数据加载和预处理
│   ├── test.ipynb            # 测试和训练的Jupyter notebook
│   ├── train_from_scratch.py # 从零开始训练
│   ├── train_with_pretrained.py # 使用预训练模型的迁移学习
│   ├── train--.py            # 简化的训练脚本
│   └── net/                  # 网络结构定义
│       ├── AlexNet.py        # AlexNet网络结构
│       ├── ResNet.py         # ResNet网络结构和残差块
│       └── ThreeLayerNetwork.py # 三层全连接网络
└── learn/                    # 学习和实验代码
    ├── 3.6.ipynb
    ├── 3.7.ipynb
    └── try.ipynb
```

## 模型架构

### ResNet18 网络结构
```
输入图像 (3×224×224)
    ↓
初始卷积层 (3×3 Conv, stride=2)
    ↓ 
ResNet Block 1 (16 channels)
    ↓
ResNet Block 2 (32 channels) 
    ↓
ResNet Block 3 (64 channels)
    ↓
ResNet Block 4 (128 channels)
    ↓
全局平均池化
    ↓
全连接层 (5 classes)
    ↓
输出预测 (5维概率分布)
```

### 三层全连接网络结构
```
输入图像 (3×224×224) 
    ↓
展平 (150,528维向量)
    ↓
全连接层1 (150,528 → 128) + ReLU
    ↓  
全连接层2 (128 → 64) + ReLU
    ↓
全连接层3 (64 → 5)
    ↓
输出预测 (5维概率分布)
```

### 训练流程图
```
数据集 (Pokemon Images)
    ↓
数据预处理 (Resize, Normalize)
    ↓
数据加载器 (DataLoader)
    ↓
┌─────────────────┐    ┌──────────────────┐
│  从头训练       │    │   迁移学习       │
│  ResNet18      │    │   预训练ResNet18  │
└─────────────────┘    └──────────────────┘
    ↓                      ↓
Adam优化器 + 交叉熵损失函数
    ↓
验证集评估 → 保存最佳模型
    ↓
测试集最终评估
```

## 主要功能

### 1. 数据处理 (`data.py`)
- **Pokemon数据集类**：自定义数据集类，支持训练/验证/测试数据划分
- **数据预处理**：图像resize、数据增强、标准化
- **数据加载**：支持多种图像格式(PNG, JPG, JPEG)
- **数据划分**：训练集60%，验证集20%，测试集20%

### 2. 网络结构 (`net/`)
- **ResNet18**：实现了标准的ResNet18网络结构
- **ThreeLayerNetwork**：简单的三层全连接网络（独立文件）
- **ResBlk**：ResNet的基本残差块实现
- **AlexNet**：经典的AlexNet网络结构

### 3. 训练方法

#### 从头训练 (`train_from_scratch.py`)
- 使用自定义ResNet18网络从零开始训练
- 支持三层全连接网络训练
- Adam优化器，交叉熵损失函数

#### 迁移学习 (`train_with_pretrained.py`)
- 使用PyTorch预训练的ResNet18模型
- 冻结特征提取层，只训练分类器
- 更快的收敛速度和更好的性能

### 4. 模型评估
- 验证集准确率监控
- 最佳模型自动保存
- 测试集性能评估

## 环境要求

### 方式一：使用requirements.txt（推荐）
```bash
pip install -r requirements.txt
```

### 方式二：手动安装
```python
torch>=1.8.0
torchvision>=0.9.0
d2l>=0.17.0
Pillow>=8.0.0
numpy>=1.19.0
matplotlib>=3.3.0
jupyter>=1.0.0
```

### 系统要求
- Python 3.7+
- CUDA 10.2+ (可选，用于GPU加速)
- 至少4GB RAM

## 快速开始

### 🚀 一键运行
```bash
# 1. 克隆项目
git clone <项目地址>
cd 3_1_project

# 2. 安装依赖
pip install torch torchvision d2l pillow numpy matplotlib

# 3. 运行训练 (推荐使用Jupyter Notebook)
jupyter notebook image_classification/test.ipynb

# 或者直接运行Python脚本
cd image_classification
python train_with_pretrained.py  # 迁移学习
python train_from_scratch.py     # 从头训练
python train--.py                # 简化版本
```

### 📊 监控训练过程
训练过程中您将看到类似输出：
```
使用设备： cuda
epoch 0, step 0, loss 1.6234
epoch 0, step 10, loss 1.4521
epoch 0, val acc 0.3421, best acc 0.3421 at epoch 0
epoch 1, step 0, loss 1.2314
epoch 1, step 10, loss 1.1203
epoch 1, val acc 0.4536, best acc 0.4536 at epoch 1
...
best acc 0.8245 at epoch 15
```

## 使用方法

### 1. 数据准备
确保Pokemon图像数据已放置在`data/pokemon/`目录下，每个类别一个子文件夹。

### 2. 训练模型

#### 使用预训练模型（推荐）
```python
# 在test.ipynb中运行或直接运行以下代码
import torch
from data import Pokemon
import train_with_pretrained as twp

# 设置训练参数
batch_size = 16
lr = 3e-4
epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_db = Pokemon('data/pokemon', 224, 'train')
val_db = Pokemon('data/pokemon', 224, 'val')
test_db = Pokemon('data/pokemon', 224, 'test')

train_iter = torch.utils.data.DataLoader(train_db, batch_size, shuffle=True, num_workers=4)
val_iter = torch.utils.data.DataLoader(val_db, batch_size, shuffle=False, num_workers=4)
test_iter = torch.utils.data.DataLoader(test_db, batch_size, shuffle=False, num_workers=4)

# 开始训练
twp.train_with_resnet18(train_iter, val_iter, epochs, lr, device)

# 测试模型
twp.test_with_resnet18(test_iter, device)
```

#### 从头训练
```python
import train_from_scratch as tfs

# 使用相同的数据加载代码
tfs.train_with_resnet18(train_iter, val_iter, epochs, lr, device)
```

### 3. 模型推理
训练完成后，最佳模型权重会自动保存为`best_resnet18.pth`，可以用于后续的推理任务。

## 技术特点

1. **模块化设计**：数据处理、网络结构、训练流程分离
2. **灵活的数据加载**：支持多种图像格式和数据增强
3. **迁移学习**：利用预训练模型提升性能
4. **自动保存**：训练过程中自动保存最佳模型
5. **设备自适应**：支持CPU和GPU训练

## 实验结果

项目提供了两种训练方式的对比：
- **从头训练**：完全训练网络所有参数
- **迁移学习**：基于ImageNet预训练模型微调

### 性能对比

| 训练方式 | 网络结构 | 训练时间 | 验证准确率 | 优势 |
|---------|----------|----------|------------|------|
| 迁移学习 | 预训练ResNet18 | 较短 | 更高 | 快速收敛，性能更好 |
| 从头训练 | 自定义ResNet18 | 较长 | 较低 | 完全掌控训练过程 |
| 简单网络 | 三层全连接 | 最短 | 最低 | 快速实验，基线对比 |

### 训练曲线示例
```
Loss曲线:
迁移学习: ████████████▓░░░░░░░  (快速下降)
从头训练: █████████████▓▓▓░░░░  (缓慢下降)
简单网络: ██████▓▓▓▓▓▓▓▓▓▓▓▓░  (收敛困难)

Accuracy曲线:
迁移学习: ░░░▓▓▓▓▓████████████  (快速提升)
从头训练: ░░░░░▓▓▓▓████████▓▓  (稳步提升)  
简单网络: ░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓  (缓慢提升)
```

迁移学习通常能够：
- ✅ 更快收敛
- ✅ 更好的分类精度  
- ✅ 更少的训练数据需求
- ✅ 更稳定的训练过程

## 扩展建议

1. **数据增强**：添加更多数据增强技术提升模型泛化能力
2. **网络结构**：尝试其他网络结构如ResNet50、EfficientNet等
3. **损失函数**：实验不同的损失函数如Focal Loss
4. **优化器**：尝试不同的优化策略
5. **集成学习**：结合多个模型提升预测性能

## 作者

Thomas

## 许可证

本项目仅供学习和研究使用。