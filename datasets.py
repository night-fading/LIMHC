import glob
import os.path
import random

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import crop

from utils import QRcode


class trainDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_train_data_path = sorted(glob.glob(os.path.join(path, 'train/*.jpg')))

    def __len__(self):
        return len(self.all_train_data_path)

    def __getitem__(self, idx):
        img_t = T.Resize(size=(256, 256))(read_image(self.all_train_data_path[idx]))
        i, j, h, w = T.RandomCrop.get_params(img_t, (96, 96))
        img_cropped_t = crop(img_t, i, j, h, w)

        message = ''.join(str(i) for i in random.choices([0, 1], k=96))
        qrcode = QRcode.QRcode().message2img(message)

        input_encoder = torch.cat((img_cropped_t, qrcode), 0)
        print(input_encoder.shape)
        return img_t, input_encoder, (i, j)


class validateDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_val_data_path = sorted(glob.glob(os.path.join(path, 'val/*.jpg')))

    def __len__(self):
        return len(self.all_val_data_path)

    def __getitem__(self, idx):
        img_t = T.Resize(size=(256, 256))(read_image(self.all_val_data_path[idx]))
        i, j, h, w = T.RandomCrop.get_params(img_t, (96, 96))
        img_cropped_t = crop(img_t, i, j, h, w)
        return img_t, img_cropped_t, (i, j)


if __name__ == "__main__":
    img_t, img_cropped_t, location = trainDataset('../data/LIMHC').__getitem__(2)
    # print(img_t.shape)
    # print(img_cropped_t.shape)
    # print(location)
    # img = T.ToPILImage()(img_cropped_t)
    # img.show()
    # img = T.ToPILImage()(img_t)
    # img.show()
