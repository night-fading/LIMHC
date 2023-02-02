import os.path

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets import dataset
from model.net import net
from utils.transforms import KeypointToHeatMap


def visualization(net, figure_path, data_path, cache_path):
    net = net()
    img_t, img_cropped_t, qrcode, input_encoder, position = iter(
        DataLoader(dataset(os.path.join(data_path, 'LIMHC'), cache_path), batch_size=1)).__next__()
    img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered = net(
        input_encoder, img_t, position)

    plt.imshow(img_entire.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_entire.png"))

    plt.imshow(img_distorted.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_distorted.png"))

    plt.imshow(img_corrected.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_corrected.png"))

    heatmap = KeypointToHeatMap()(pos)
    plt.imshow(heatmap.cpu().detach().squeeze(0)[0])
    plt.savefig(os.path.join(figure_path, "heatmap0.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[1])
    plt.savefig(os.path.join(figure_path, "heatmap1.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[2])
    plt.savefig(os.path.join(figure_path, "heatmap2.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[3])
    plt.savefig(os.path.join(figure_path, "heatmap3.png"))

    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[0])
    plt.savefig(os.path.join(figure_path, "heatmap_pred0.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[1])
    plt.savefig(os.path.join(figure_path, "heatmap_pred1.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[2])
    plt.savefig(os.path.join(figure_path, "heatmap_pred2.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[3])
    plt.savefig(os.path.join(figure_path, "heatmap_pred3.png"))

    plt.imshow(qrcode_recovered.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "qrcode_recovered.png"))


if __name__ == "__main__":
    visualization(net, "../figure", "../../data", "../.cache")
