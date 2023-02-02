from torch import nn
from torch.utils.data import DataLoader

from datasets import dataset
from model.decoder import decoder
from model.encoder import encoder
from model.hrnet import HighResolutionNet
from utils.transforms import replaceImage, distort, get_max_preds, correctSubImage


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.hrnet = HighResolutionNet()
        self.decoder = decoder()

    def forward(self, input_t, img_cover, pos):
        img_encoded = self.encoder(input_t)
        img_entire = replaceImage(img_cover, img_encoded, pos)
        img_distorted, pos = distort(img_entire, pos)
        heatmap_pred = self.hrnet(img_distorted.clone())
        pos_pred, max_vals = get_max_preds(heatmap_pred)
        img_corrected = correctSubImage(img_distorted, pos_pred * 4)
        qrcode_recovered = self.decoder(img_corrected)
        return img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered


if __name__ == "__main__":
    net = net()
    img_t, img_cropped_t, qrcode, input_encoder, position = iter(
        DataLoader(dataset('../../data/LIMHC', '../.cache'), batch_size=2)).__next__()
    net(input_encoder, img_t, position)
