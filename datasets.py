import glob
import os.path
import random

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import crop

from utils.QRcode import QRcode


class dataset(Dataset):
    def __init__(self, data_path, cache_path):
        self.data_path = data_path
        self.cache_path = cache_path
        self.all_train_data_path = sorted(glob.glob(os.path.join(data_path, 'train/*.jpg')))

    def __len__(self):
        return len(self.all_train_data_path)

    def __getitem__(self, idx):
        img_t = T.Resize(size=(256, 256))(read_image(self.all_train_data_path[idx]).to(torch.float32) / 255)

        i, j, h, w = T.RandomCrop.get_params(img_t, (96, 96))
        img_cropped_t = crop(img_t, i, j, h, w)

        message = ''.join(str(i) for i in random.choices([0, 1], k=96))
        qrcode = QRcode(self.cache_path).message2img(message)

        input_encoder = torch.cat((img_cropped_t, qrcode), 0)
        pos = torch.tensor([(i, j), (i, j + 96), (i + 96, j + 96), (i + 96, j)])
        return img_t, img_cropped_t, qrcode, input_encoder, pos
