import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from matplotlib import pyplot as plt

from datasets import trainDataset
from model.unet import UNet


def replaceImage(replaced_image, image, position):
    i, j = position[0]
    replaced_image[:, i: i + 96, j:j + 96] = image
    return replaced_image


def distort(img, position):
    # print(img)
    img_d = img.detach()
    plt.imshow(img_d.permute(1, 2, 0))
    plt.show()

    img = T.ColorJitter(.5, .5, .5, .1)(img)
    img = T.GaussianBlur(3, (0.1, 1))(img)
    img, pos = perspectiveTransform(img, position)

    return img, pos


def perspectiveTransform(img, position):
    pts1, pts2 = T.RandomPerspective().get_params(256, 256, 0.5)
    img = F.perspective(img, pts1, pts2)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    M = cv.getPerspectiveTransform(pts1, pts2)

    position[0] = cvt_pos((position[0][1], position[0][0]), M)
    position[1] = cvt_pos((position[1][1], position[1][0]), M)
    position[2] = cvt_pos((position[2][1], position[2][0]), M)
    position[3] = cvt_pos((position[3][1], position[3][0]), M)

    # img = (img * 255).to(torch.uint8)
    # pos = torch.tensor([[position[3][0], position[3][1], position[3][0]+1, position[3][1]+1]], dtype=torch.float)
    # img = draw_bounding_boxes(img, pos, colors=["blue"], width=5)

    return img, position


def cvt_pos(position, cvt_mat_t):
    u, v = position
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    return (int(x), int(y))


if __name__ == "__main__":
    img_t, img_cropped_t, position = trainDataset('../../data/LIMHC', '../.cache').__getitem__(3)
    net = UNet()
    img_cropped_t = img_cropped_t.unsqueeze(0)
    x = net.forward(img_cropped_t)
    x = torch.clamp(x, min=0.0, max=1.0)
    # print((x > 1.0).any())
    x = replaceImage(img_t, x, position=position)
    x, pos = distort(x, position)
    # x = perspectiveTransform(x, position)
    print(type(x))
    x_d = x.detach()
    plt.imshow(x_d.permute(1, 2, 0))
    plt.show()
