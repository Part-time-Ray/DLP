import torch
import torch.nn as nn
from diffusers import UNet2DModel

class UNet(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.dim = 512
        self.cond_embed = nn.Linear(num_classes, self.dim)
        self.model = UNet2DModel(
            sample_size=64,
            in_channels=3 + self.dim,
            out_channels=3,
            layers_per_block=3,
            block_out_channels=[128, 128, 256, 256, 512, 512],
            down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
            up_block_types=["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"]
        )
    def forward(self, x, label, t):
        label = label.float()
        batch = label.shape[0]
        cond = self.cond_embed(label).view(batch, self.dim, 1, 1).expand(batch, self.dim, 64, 64)
        x = torch.cat((x, cond), dim=1)
        
        label = label.float()
        return self.model(x, t).sample
    
if __name__ == "__main__":
    model = UNet()
    print(model)
    print(model(torch.randn(1, 3, 64, 64), torch.randint(0, 1, (1, 24), dtype=torch.float)).shape, 10)