import os.path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import dataset
from model.net import net
from utils.transforms import KeypointToHeatMap


def visualization(net, figure_path, data_path, cache_path):
    data_iter = iter(DataLoader(dataset(os.path.join(data_path, 'LIMHC'), cache_path), batch_size=1))
    img_t, img_cropped_t, qrcode, input_encoder, position = data_iter.__next__()
    img_t, img_cropped_t, qrcode, input_encoder, position = img_t.to(
        'cuda' if torch.cuda.is_available() else 'cpu'), img_cropped_t.to(
        'cuda' if torch.cuda.is_available() else 'cpu'), qrcode.to(
        'cuda' if torch.cuda.is_available() else 'cpu'), input_encoder.to(
        'cuda' if torch.cuda.is_available() else 'cpu'), position.to(
        'cuda' if torch.cuda.is_available() else 'cpu')

    img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered = net(
        input_encoder, img_t, position)

    plt.imshow(img_entire.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_entire.png"))

    plt.imshow(img_distorted.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_distorted.png"))

    plt.imshow(img_corrected.cpu().detach().squeeze(0).permute(1, 2, 0))
    plt.savefig(os.path.join(figure_path, "img_corrected.png"))

    heatmap = KeypointToHeatMap()(pos)
    plt.imshow(heatmap.cpu().detach().squeeze(0)[0], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap0.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[1], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap1.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[2], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap2.png"))
    plt.imshow(heatmap.cpu().detach().squeeze(0)[3], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap3.png"))

    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[0], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap_pred0.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[1], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap_pred1.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[2], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap_pred2.png"))
    plt.imshow(heatmap_pred.cpu().detach().squeeze(0)[3], plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "heatmap_pred3.png"))

    plt.imshow(qrcode.cpu().detach().squeeze(0).squeeze(0), plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "qrcode.png"))
    plt.imshow(qrcode_recovered.cpu().detach().squeeze(0).permute(1, 2, 0), plt.cm.gray)
    plt.savefig(os.path.join(figure_path, "qrcode_recovered.png"))

    plt.clf()


if __name__ == "__main__":
    net = net().to('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load('../5', map_location='cpu'))
    visualization(net, "../figure", "../../data", "../.cache")
