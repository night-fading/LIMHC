import torch
from matplotlib import pyplot as plt

from datasets import trainDataset
from model.unet import UNet


def replaceImage(replaced_image, image, position):
    print(replaced_image.shape)

    i, j = position
    replaced_image[:, i: i + 96, j:j + 96] = image
    print(replaced_image.shape)
    plt.imshow(replaced_image.to(torch.uint8).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    img_t, img_cropped_t, location = trainDataset('../../data/LIMHC', '../.cache').__getitem__(4)
    net = UNet()
    img_cropped_t = img_cropped_t.unsqueeze(0)
    # img = T.ToPILImage()(img_t)
    # img.show()
    # print(img_cropped_t.shape)
    x = net.forward(img_cropped_t)
    x = x.squeeze(0)

    replaceImage(img_t, x, position=location)
