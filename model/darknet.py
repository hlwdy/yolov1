import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, layers_param, use_pool=True):
        super().__init__()
        layers = []
        for in_ch, out_ch, k_size, stride, padding in layers_param:
            layers.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, bias=False))
            layers.append(nn.LeakyReLU(0.1))
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2) if use_pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()

        self.conv1 = Block([[3, 64, 7, 2, 3]])
        self.conv2 = Block([[64, 192, 3, 1, 1]])
        self.conv3 = Block([
            [192, 128, 1, 1, 0],
            [128, 256, 3, 1, 1],
            [256, 256, 1, 1, 0],
            [256, 512, 3, 1, 1]
        ])
        self.conv4 = Block([
            [512, 256, 1, 1, 0],
            [256, 512, 3, 1, 1],
            [512, 256, 1, 1, 0],
            [256, 512, 3, 1, 1],
            [512, 256, 1, 1, 0],
            [256, 512, 3, 1, 1],
            [512, 256, 1, 1, 0],
            [256, 512, 3, 1, 1],
            [512, 512, 1, 1, 0],
            [512, 1024, 3, 1, 1]
        ])
        self.conv5 = Block([
            [1024, 512, 1, 1, 0],
            [512, 1024, 3, 1, 1],
            [1024, 512, 1, 1, 0],
            [512, 1024, 3, 1, 1],
            [1024, 1024, 3, 1, 1],
            [1024, 1024, 3, 2, 1]
        ], use_pool=False)
        self.conv6 = Block([
            [1024, 1024, 3, 1, 1],
            [1024, 1024, 3, 1, 1]
        ], use_pool=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


if __name__ == "__main__":
    x = torch.randn(1, 3, 448, 448)
    net = DarkNet()
    print(net)
    out = net(x)
    print(out.shape)  # 输出: torch.Size([1, 1024, 7, 7])