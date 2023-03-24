import torch
import torch.nn as nn

class Estimation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(Estimation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)
        self.conv = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3,
                              stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        x = self.conv(inputs)      # [N, 1, D, H, W]
        x = torch.squeeze(x, dim=-1)  # [N, 1, D, H, W]
        x = x.permute(0, 3, 4, 2, 1)  # [N, H, W, D, 1]
        assert x.shape[-2] == self.max_disp - self.min_disp
        candidates = torch.linspace(1.0 * self.min_disp, 1.0 * self.max_disp - 1.0, self.max_disp - self.min_disp).to(x.device)
        probabilities = self.softmax(-1.0 * x)  # [N, H, W, D, 1]
        disparities = torch.sum(candidates * probabilities, dim=-2, keepdim=True)  # [N, H, W, 1]
        return disparities
