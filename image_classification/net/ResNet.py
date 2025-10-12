import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlk(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):# stride控制是否降采样
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        # 用于匹配通道和尺寸
        self.extra = nn.Sequential()
        if ch_out != ch_in or stride != 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.extra(x) # 残差边
        out = F.relu(out)
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # 各阶段残差块
        self.blk1 = ResBlk(16, 32, stride=2)
        self.blk2 = ResBlk(32, 64, stride=2)
        self.blk3 = ResBlk(64, 128, stride=2)
        self.blk4 = ResBlk(128, 256, stride=2)
        self.outlayer = nn.Linear(256, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # 全局平均池化
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x



if __name__ == '__main__':
    from ThreeLayerNetwork import Three_Layer_Network
    
    model = Three_Layer_Network()
    print(model)
    sample_input = torch.randn(2, 3, 224, 224)
    sample_output = model(sample_input)
    print(sample_output.shape)  # should be [2, 5]

    blk = ResBlk(64, 128, stride=2)
    tmp = torch.randn(2, 64, 224, 224)
    out = blk(tmp)
    print(out.shape)

    model = ResNet18()
    tmp = torch.randn(2, 3, 224, 224)
    out = model(tmp)
    print('model:', out.shape)
