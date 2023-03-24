import torch
import torch.nn as nn

class CostConcatenation(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(CostConcatenation, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs):
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(nn.functional.pad(
                    input=torch.cat([inputs[0][:, :, :i, :], inputs[1][:, :, -i:, :]], dim=-1),
                    pad=(0, 0, 0, -i),
                    mode='constant'))
            elif i > 0:
                cost_volume.append(nn.functional.pad(
                    input=torch.cat([inputs[0][:, :, i:, :], inputs[1][:, :, :-i, :]], dim=-1),
                    pad=(i, 0, 0, 0),
                    mode='constant'))
            else:
                cost_volume.append(torch.cat([inputs[0], inputs[1]], dim=-1))
        cost_volume = torch.stack(cost_volume, dim=1)
        return cost_volume

class CostDifference(nn.Module):
    def __init__(self, min_disp=-112.0, max_disp=16.0):
        super(CostDifference, self).__init__()
        self.min_disp = int(min_disp)
        self.max_disp = int(max_disp)

    def forward(self, inputs):
        assert len(inputs) == 2
        cost_volume = []
        for i in range(self.min_disp, self.max_disp):
            if i < 0:
                cost_volume.append(nn.functional.pad(
                    input=inputs[0][:, :, :i, :] - inputs[1][:, :, -i:, :],
                    pad=(0, 0, 0, -i),
                    mode='constant'))
            elif i > 0:
                cost_volume.append(nn.functional.pad(
                    input=inputs[0][:, :, i:, :] - inputs[1][:, :, :-i, :],
                    pad=(i, 0, 0, 0),
                    mode='constant'))
            else:
                cost_volume.append(inputs[0] - inputs[1])
        cost_volume = torch.stack(cost_volume, dim=1)
        return cost_volume
