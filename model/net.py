import torch
from torch import nn

from model.decoder import decoder
from model.encoder import encoder
from model.hrnet import HighResolutionNet
from utils.transforms import replaceImage, distort, correctSubImage, get_final_preds


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.hrnet = HighResolutionNet()
        self.decoder = decoder()

    def forward(self, input_t, img_cover, pos):
        img_encoded = self.encoder(input_t)
        img_encoded = torch.clamp(img_encoded.clone(), min=0.0, max=1.0)
        img_entire = replaceImage(img_cover.clone(), img_encoded, pos)
        img_distorted, pos = distort(img_entire, pos)
        img_distorted = torch.clamp(img_distorted.clone(), min=0.0, max=1.0)
        heatmap_pred = self.hrnet(img_distorted.clone())
        pos_pred, max_vals = get_final_preds(heatmap_pred)
        img_corrected = correctSubImage(img_distorted.clone(), pos_pred)
        qrcode_recovered = self.decoder(img_corrected)
        qrcode_recovered = torch.clamp(qrcode_recovered.clone(), min=0.0, max=1.0)
        return img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered
