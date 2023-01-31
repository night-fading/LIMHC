import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def replaceImage(replaced_image, image, position):
    i, j = position[0]
    replaced_image[:, i: i + 96, j:j + 96] = image
    return replaced_image


def distort(img, position):
    img = T.ColorJitter(.5, .5, .5, .1)(img)
    img = T.GaussianBlur(3, (0.1, 1))(img)
    img, position_transformed = perspectiveTransform(img, position)

    return img, position_transformed


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
    return img, position


def cvt_pos(position, cvt_mat_t):
    u, v = position
    x = (cvt_mat_t[0][0] * u + cvt_mat_t[0][1] * v + cvt_mat_t[0][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0] * u + cvt_mat_t[1][1] * v + cvt_mat_t[1][2]) / (
            cvt_mat_t[2][0] * u + cvt_mat_t[2][1] * v + cvt_mat_t[2][2])
    return int(x), int(y)


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, torch.Tensor), 'batch_heatmaps should be torch.Tensor'
    assert len(batch_heatmaps.shape) == 4, 'batch_images should be 4-ndim'

    batch_size, num_joints, h, w = batch_heatmaps.shape
    heatmaps_reshaped = batch_heatmaps.reshape(batch_size, num_joints, -1)
    maxvals, idx = torch.max(heatmaps_reshaped, dim=2)
    maxvals = maxvals.unsqueeze(dim=-1)
    idx = idx.float()

    preds = torch.zeros((batch_size, num_joints, 2)).to(batch_heatmaps)

    preds[:, :, 0] = idx % w  # column 对应最大值的x坐标
    preds[:, :, 1] = torch.floor(idx / w)  # row 对应最大值的y坐标

    pred_mask = torch.gt(maxvals, 0.0).repeat(1, 1, 2).float().to(batch_heatmaps.device)

    preds *= pred_mask
    return preds, maxvals


def correctSubImage(img, pos):
    pts1 = pos
    pts2 = [[0, 0], [96, 0], [96, 96], [0, 96]]

    img = F.perspective(img, pts1, pts2)
    img = img[:, :96, :96]
    return img
