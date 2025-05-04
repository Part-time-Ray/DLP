import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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
    
class DOWN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DOWN, self).__init__()
        self.conv = CC(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        output = self.conv(x)
        return self.pool(output), output

class UP(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UP, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = CC(in_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
        


class UNet(nn.Module):
    def __init__(self, time_step, in_channels = 3, out_channels = 3, num_classes = 24):
        super(UNet, self).__init__()
        self.down1 = DOWN(in_channels, 64)
        self.down2 = DOWN(64 * 2, 128)
        self.down3 = DOWN(128 * 2, 256)
        self.down4 = DOWN(256 * 2, 512)
        self.bottleneck = CC(512 * 2, 512)
        self.up4 = UP(512 * 2, 256, skip_channels=512)
        self.up3 = UP(256 * 2, 128, skip_channels=256)
        self.up2 = UP(128 * 2, 64, skip_channels=128)
        self.up1 = UP(64 * 2, out_channels, skip_channels=64)
        self.final_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

        self.dim = 128
        self.cond_embed = nn.Sequential(
            nn.Embedding(num_classes, self.dim),
            nn.Linear(self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, self.dim)
        )
        self.time_embed = nn.Parameter(self.sinusoidal_embedding(time_step, self.dim), requires_grad=False)
        self.boardcast_down1 = nn.Linear(self.dim * 2, 64)
        self.boardcast_down2 = nn.Linear(self.dim * 2, 128)
        self.boardcast_down3 = nn.Linear(self.dim * 2, 256)
        self.boardcast_down4 = nn.Linear(self.dim * 2, 512)
        self.boardcast_bottleneck = nn.Linear(self.dim * 2, 512)
        self.boardcast_up4 = nn.Linear(self.dim * 2, 256)
        self.boardcast_up3 = nn.Linear(self.dim * 2, 128)
        self.boardcast_up2 = nn.Linear(self.dim * 2, 64)
        self.boardcast_up1 = nn.Linear(self.dim * 2, out_channels)
    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(timesteps, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = torch.pad(emb, (0, 1, 0, 0))
        return emb
    def forward(self, x, cond, t):
        cond_embed = torch.matmul(cond.float(), self.cond_embed[0].weight)
        cond_embed = self.cond_embed[1:](cond_embed)
        t_embed = self.time_embed[t]

        embed = torch.cat([cond_embed, t_embed], dim=1)

        x1, o1 = self.down1(x)
        embed1 = self.boardcast_down1(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x1.size(2), x1.size(3))
        x1 = torch.cat([x1, embed1], dim=1)

        x2, o2 = self.down2(x1)
        embed2 = self.boardcast_down2(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x2.size(2), x2.size(3))
        x2 = torch.cat([x2, embed2], dim=1)

        x3, o3 = self.down3(x2)
        embed3 = self.boardcast_down3(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x3.size(2), x3.size(3))
        x3 = torch.cat([x3, embed3], dim=1)

        x4, o4 = self.down4(x3)
        embed4 = self.boardcast_down4(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x4.size(2), x4.size(3))
        x4 = torch.cat([x4, embed4], dim=1)

        x5 = self.bottleneck(x4)
        embed5 = self.boardcast_bottleneck(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x5.size(2), x5.size(3))
        x5 = torch.cat([x5, embed5], dim=1)

        x6 = self.up4(x5, o4)
        embed6 = self.boardcast_up4(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x6.size(2), x6.size(3))
        x6 = torch.cat([x6, embed6], dim=1)

        x7 = self.up3(x6, o3)
        embed7 = self.boardcast_up3(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x7.size(2), x7.size(3))
        x7 = torch.cat([x7, embed7], dim=1)

        x8 = self.up2(x7, o2)
        embed8 = self.boardcast_up2(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x8.size(2), x8.size(3))
        x8 = torch.cat([x8, embed8], dim=1)

        x9 = self.up1(x8, o1)
        embed9 = self.boardcast_up1(embed).view(embed.size(0), -1, 1, 1).expand(-1, -1, x9.size(2), x9.size(3))
        x9 = torch.cat([x9, embed9], dim=1)

        return self.final_conv(x9)
        

if __name__ == "__main__":  
    model = UNet(in_channels=3, out_channels=3)
    input_tensor = torch.randn(1, 3, 64, 64)
    label = [0, 5, 23]
    multihot = torch.zeros(1, 24)
    multihot[0, label] = 1
    output_tensor = model(input_tensor, cond = multihot.detach().clone(), t=torch.tensor([20]))

    print("input size:", input_tensor.shape)
    print("output size:", output_tensor.shape)