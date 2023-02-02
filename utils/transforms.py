from typing import Tuple

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F


def replaceImage(img_cover, sub_img, position):
    batch_size = img_cover.shape[0]
    for k in range(batch_size):
        i, j = position[k][0]
        img_cover[k, :, i: i + 96, j:j + 96] = sub_img[k].clone()
    return img_cover


def distort(img, position):
    img = T.ColorJitter(.5, .5, .5, .1)(img)
    img = T.GaussianBlur(3, (0.1, 1))(img)

    batch_size = img.shape[0]
    for k in range(batch_size):
        # if random.choices([0, 1])[0]:
        if 0:
            img[k], position[k] = perspectiveTransform(img[k].clone(), position[k].clone())
        else:
            position[k][0] = torch.tensor([position[k][0][1], position[k][0][0]])
            position[k][1] = torch.tensor([position[k][1][1], position[k][1][0]])
            position[k][2] = torch.tensor([position[k][2][1], position[k][2][0]])
            position[k][3] = torch.tensor([position[k][3][1], position[k][3][0]])
    return img, position


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
    return torch.tensor([int(x), int(y)])


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
    batch_size = img.shape[0]
    for k in range(batch_size):
        pts1 = pos[k].tolist()
        pts2 = [[0, 0], [96, 0], [96, 96], [0, 96]]
        img[k] = F.perspective(img[k].clone(), pts1, pts2)
    img = img[:, :, :96, :96]
    return img


class KeypointToHeatMap(object):
    def __init__(self,
                 heatmap_hw: Tuple[int, int] = (256 // 4, 256 // 4),
                 gaussian_sigma: int = 2,
                 keypoints_weights=None):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel_radius = self.sigma * 3
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

        # generate gaussian kernel(not normalized)
        kernel_size = 2 * self.kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        x_center = y_center = kernel_size // 2
        for x in range(kernel_size):
            for y in range(kernel_size):
                kernel[y, x] = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * self.sigma ** 2))
        # print(kernel)

        self.kernel = kernel

    def compute(self, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)
        if "visible" in target:
            visible = target["visible"]
            kps_weights = visible

        heatmap = np.zeros((num_kps, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).numpy().astype(np.int32)  # round
        for kp_id in range(num_kps):
            v = kps_weights[kp_id]
            if v < 0.5:
                # 如果该点的可见度很低，则直接忽略
                continue

            x, y = heatmap_kps[kp_id]
            ul = [x - self.kernel_radius, y - self.kernel_radius]  # up-left x,y
            br = [x + self.kernel_radius, y + self.kernel_radius]  # bottom-right x,y
            # 如果以xy为中心kernel_radius为半径的辐射范围内与heatmap没交集，则忽略该点(该规则并不严格)
            if ul[0] > self.heatmap_hw[1] - 1 or \
                    ul[1] > self.heatmap_hw[0] - 1 or \
                    br[0] < 0 or \
                    br[1] < 0:
                # If not, just return the image as is
                kps_weights[kp_id] = 0
                continue

            # Usable gaussian range
            # 计算高斯核有效区域（高斯核坐标系）
            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            # image range
            # 计算heatmap中的有效区域（heatmap坐标系）
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            if kps_weights[kp_id] > 0.5:
                # 将高斯核有效区域复制到heatmap对应区域
                heatmap[kp_id][img_y[0]:img_y[1] + 1, img_x[0]:img_x[1] + 1] = \
                    self.kernel[g_y[0]:g_y[1] + 1, g_x[0]:g_x[1] + 1]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        # plot_heatmap(image, heatmap, kps, kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)

        return target

    def __call__(self, target):
        batch_size = target.shape[0]
        heatmap = torch.zeros(batch_size, 4, 64, 64)
        for k in range(batch_size):
            kps = {"keypoints": target[k]}
            heatmap[k] = self.compute(kps)["heatmap"]
        return heatmap