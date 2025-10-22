import torch.nn as nn
import torch.nn.functional as F

class CNNSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNSmall, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 假设输入是32x32的图像
        # 经过两次2x2池化后，尺寸变为 32 -> 16 -> 8
        # 所以进入全连接层的特征图大小是 8x8
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc = nn.Linear(128, num_classes) # 使用 num_classes 参数

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8) # 扁平化
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        return x

def Model(num_classes=10):
    """
    一个符合您项目接口的工厂函数。
    它现在可以接收 num_classes 参数。
    """
    model = CNNSmall(num_classes=num_classes)
    # 遵循您代码的返回格式 (model, head)
    return model, model.fc

if __name__ == '__main__':
    model, _ = Model(num_classes=10)
    print(model)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_param}")