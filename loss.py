import torch
from torch import nn
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.transforms import KeypointToHeatMap


class loss:
    def step1(self, pos, heatmap_perd, qrcode, qrcode_recovered, img_encoded, img_cropped_t, img_t, img_entire):
        heatmap_ground_truth = KeypointToHeatMap()(pos).to('cuda' if torch.cuda.is_available() else 'cpu')
        img_t = img_t.to('cuda' if torch.cuda.is_available() else 'cpu')
        qrcode = qrcode.to('cuda' if torch.cuda.is_available() else 'cpu')

        loss1 = nn.MSELoss()
        loss2 = nn.MSELoss()
        lpips = LearnedPerceptualImagePatchSimilarity('squeeze', normalize=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        ssim = StructuralSimilarityIndexMeasure().to('cuda' if torch.cuda.is_available() else 'cpu')

        part1 = loss1(heatmap_ground_truth, heatmap_perd)
        part2 = lpips(img_t, img_entire)
        part3 = loss2(qrcode, qrcode_recovered)
        return part1 + 5 * part2 + part3
