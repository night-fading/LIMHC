import glob
import os.path

import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image


class trainDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_train_data_path = sorted(glob.glob(os.path.join(path, 'train/*.jpg')))

    def __len__(self):
        return len(self.all_train_data_path)

    def __getitem__(self, idx):
        return read_image(self.all_train_data_path[idx])


class validateDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_val_data_path = sorted(glob.glob(os.path.join(path, 'val/*.jpg')))

    def __len__(self):
        return len(self.all_val_data_path)

    def __getitem__(self, idx):
        print(self.all_val_data_path[idx])
        return read_image(self.all_val_data_path[idx])


if __name__ == "__main__":
    img_t = validateDataset('../data/LIMHC').__getitem__(2)
    print(img_t.shape)
    img = T.ToPILImage()(img_t)
    img.show()
