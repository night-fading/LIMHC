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

    def forward(self, input_t, img_cover, pos, img_cropped_t):
        img_encoded = self.encoder(input_t)
        # visualization(img_encoded.detach().squeeze(0).permute(1, 2, 0))
        img_entire = replaceImage(img_cover, img_encoded, pos)
        # visualization(img_entire[1].detach().squeeze(0).permute(1, 2, 0))
        img_distorted, pos = distort(img_entire, pos)
        # visualization(img_distorted[0].detach().squeeze(0).permute(1, 2, 0))
        # visualization(img_distorted[1].detach().squeeze(0).permute(1, 2, 0))
        pos_pred, max_vals = get_max_preds(self.hrnet(img_distorted))
        img_corrected = correctSubImage(img_distorted, pos)
        # visualization(img_corrected[0].detach().squeeze(0).permute(1, 2, 0))
        # visualization(img_corrected[1].detach().squeeze(0).permute(1, 2, 0))
        qrcode_recovered = self.decoder(img_corrected)
        # visualization(qrcode_recovered[1].detach().squeeze(0))


if __name__ == "__main__":
    net = net()
    img_t, img_cropped_t, qrcode, input_encoder, position = iter(
        DataLoader(dataset('../../data/LIMHC', '../.cache'), batch_size=2)).__next__()
    net(input_encoder, img_t, position, img_cropped_t)
