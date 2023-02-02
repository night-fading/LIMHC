import torch
from torch import nn

from utils.transforms import KeypointToHeatMap


class loss:
    def step1(self, pos, heatmap_perd, qrcode, qrcode_recovered):
        loss1 = nn.MSELoss()
        loss2 = nn.CrossEntropyLoss()
        heatmap_ground_truth = KeypointToHeatMap()(pos)
        return 30 * loss1(heatmap_ground_truth.to('cuda' if torch.cuda.is_available() else 'cpu'), heatmap_perd)
        # + loss2(qrcode, qrcode_recovered)
