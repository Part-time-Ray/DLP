import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn

class CC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class ResNet34_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNet34_UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.encoder.append(self.add_layer(BasicBlock, 64, 64, 3))   # Layer 1
        self.encoder.append(self.add_layer(BasicBlock, 64, 128, 4, stride=2))  # Layer 2
        self.encoder.append(self.add_layer(BasicBlock, 128, 256, 6, stride=2)) # Layer 3
        self.encoder.append(self.add_layer(BasicBlock, 256, 512, 3, stride=2)) # Layer 4
        self.bottleneck = CC(512, 256)

        self.decoder = nn.ModuleList()
        self.decoder.append(self.add_up(256 + 512, 128, scale_factor=2))
        self.decoder.append(self.add_up(128 + 256, 64, scale_factor=2))
        self.decoder.append(self.add_up(64 + 128, 32, scale_factor=2))
        self.decoder.append(self.add_up(32 + 64, 16, scale_factor=2))
        
        self.upConv = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.re = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        skip = []
        x = self.conv(x)
        x = self.pool(x)
        for layer in self.encoder:
            x = layer(x)
            skip.append(x)
        x = self.bottleneck(x)
        skip = skip[::-1]
        for idx in range(len(self.decoder)):
            skip_connection = skip[idx]
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)
        x = self.upConv(x)
        return self.re(x)

    def add_layer(self, block, in_channels, out_channels, num, stride=1):
        layers = [block(in_channels, out_channels, stride=stride)]
        for _ in range(num - 1):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def add_up(self, in_channels, out_channels, scale_factor=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor),
            CC(out_channels, out_channels)
        )

if __name__ == "__main__":  
    model = ResNet34_UNet(in_channels=3, out_channels=1)
    input_tensor = torch.randn(1, 3, 256,256)
    output_tensor = model(input_tensor)

    print("input size:", input_tensor.shape)
    print("output size:", output_tensor.shape)


        