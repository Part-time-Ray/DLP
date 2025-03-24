import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class CC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CC, self).__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[4, 8, 16, 32]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.encoder.append(CC(in_channels, feature))
            in_channels = feature
        self.bottleneck = CC(features[-1], features[-1]*2)
        features.reverse()
        for feature in features:
            self.decoder.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.decoder.append(CC(feature*2, feature))
        self.re = nn.Conv2d(features[-1], out_channels, kernel_size=1)
    def forward(self, x):
        skip = []
        for layer in self.encoder:
            x = layer(x)
            skip.append(x)
            x = self.pool(x)
            # print('encoder:', x.shape)
        x = self.bottleneck(x)
        # print('bottleneck:', x.shape)
        skip = skip[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip[idx//2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx+1](concat_skip)
            # print('decoder:', x.shape)
        return self.re(x)

if __name__ == "__main__":  
    model = UNet(in_channels=3, out_channels=1)
    input_tensor = torch.randn(1, 3, 256, 256)
    output_tensor = model(input_tensor)

    print("input size:", input_tensor.shape)
    print("output size:", output_tensor.shape)