import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.transforms import KeypointToHeatMap


class loss:
    def step1(self, pos, heatmap_perd, qrcode, qrcode_recovered, img_encoded, img_cropped_t):
        heatmap_ground_truth = KeypointToHeatMap()(pos).to('cuda' if torch.cuda.is_available() else 'cpu')
        img_cropped_t = img_cropped_t.to('cuda' if torch.cuda.is_available() else 'cpu')
        qrcode = qrcode.to('cuda' if torch.cuda.is_available() else 'cpu')

        loss1 = nn.MSELoss()
        loss3 = nn.MSELoss()
        lpips = LearnedPerceptualImagePatchSimilarity().to('cuda' if torch.cuda.is_available() else 'cpu')

        part1 = loss1(heatmap_ground_truth, heatmap_perd)
        part2 = lpips(img_cropped_t, img_encoded)
        part3 = loss3(qrcode, qrcode_recovered)
        return part1 + 15 * part2 + 1e-1 * part3
