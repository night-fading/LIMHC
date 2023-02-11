import random

import torch
import torchvision.transforms as T
from torch import nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import crop

from model.decoder import decoder
from model.encoder import encoder
from model.hrnet import HighResolutionNet
from utils.QRcode import QRcode
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
        img_distorted, pos = distort(img_entire.clone(), pos)
        img_distorted = torch.clamp(img_distorted.clone(), min=0.0, max=1.0)
        heatmap_pred = self.hrnet(img_distorted.clone())
        pos_pred, max_vals = get_final_preds(heatmap_pred)
        img_corrected = correctSubImage(img_distorted.clone(), pos_pred)
        qrcode_recovered = self.decoder(img_corrected)
        qrcode_recovered = torch.clamp(qrcode_recovered.clone(), min=0.0, max=1.0)
        return img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered

    def encode(self, img_pth):
        img_t = T.Resize(size=(256, 256))(read_image(img_pth).to(torch.float32) / 255)

        i, j, h, w = T.RandomCrop.get_params(img_t, (96, 96))
        pos = torch.tensor([(i, j), (i, j + 96), (i + 96, j + 96), (i + 96, j)])
        img_cropped_t = crop(img_t, i, j, h, w)

        message = ''.join(str(i) for i in random.choices([0, 1], k=96))
        qrcode = QRcode("../.cache").message2img(message)

        input_encoder = torch.cat((img_cropped_t, qrcode), 0)

        img_encoded = self.encoder(input_encoder.unsqueeze(0))
        img_encoded = torch.clamp(img_encoded.clone(), min=0.0, max=1.0)
        img_entire = replaceImage(img_t.unsqueeze(0).clone(), img_encoded, pos.unsqueeze(0))

        img = T.ToPILImage()(img_entire.squeeze(0))
        img.save(img_pth[:-4] + "_encoded.png")

    def decode(self, img_pth):
        img_t = T.Resize(size=(256, 256))(read_image(img_pth, mode=ImageReadMode.RGB).to(torch.float32) / 255)
        heatmap_pred = self.hrnet(img_t.unsqueeze(0).clone())
        pos_pred, max_vals = get_final_preds(heatmap_pred)
        img_corrected = correctSubImage(img_t.unsqueeze(0).clone(), pos_pred)
        qrcode_recovered = self.decoder(img_corrected)
        qrcode_recovered = torch.clamp(qrcode_recovered.clone(), min=0.0, max=1.0)

        img = T.ToPILImage()(qrcode_recovered.squeeze(0).squeeze(0))
        img.save(img_pth[:-4] + "_recovered.png")


if __name__ == "__main__":
    net = net().to('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    net.load_state_dict(torch.load('../model_weights.pth', map_location='cuda' if torch.cuda.is_available() else 'cpu'))

    # net.encode("/Users/leizhe/fsdownload/2007_007948.jpg")
    net.decode("/Users/leizhe/fsdownload/8.png")
