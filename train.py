import datetime
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from datasets import dataset
from loss import loss
from model.net import net
from utils.visualization import visualization


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, net, loss_fn,
                 stage="train", metrics_dict=None,
                 optimizer=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer = optimizer

    def step(self, batch):
        img_t, img_cropped_t, qrcode, input_encoder, position = batch
        img_t, img_cropped_t, qrcode, input_encoder, position = img_t.to(
            'cuda' if torch.cuda.is_available() else 'cpu'), img_cropped_t.to(
            'cuda' if torch.cuda.is_available() else 'cpu'), qrcode.to(
            'cuda' if torch.cuda.is_available() else 'cpu'), input_encoder.to(
            'cuda' if torch.cuda.is_available() else 'cpu'), position.to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # loss
        img_encoded, img_entire, img_distorted, pos, heatmap_pred, pos_pred, max_vals, img_corrected, qrcode_recovered = self.net(
            input_encoder, img_t, position)

        loss = loss_fn(pos, heatmap_pred, qrcode, qrcode_recovered, img_encoded, img_cropped_t, img_t, img_entire)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(img_encoded, img_cropped_t).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics

    def train_step(self, batch):
        self.net.train()  # 训练模式, dropout层发生作用
        return self.step(batch)

    @torch.no_grad()
    def eval_step(self, batch):
        self.net.eval()  # 预测模式, dropout层不发生作用
        return self.step(batch)

    def __call__(self, batch):
        if self.stage == "train":
            return self.train_step(batch)
        else:
            return self.eval_step(batch)


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader), file=sys.stdout)
        for i, batch in loop:
            loss, step_metrics = self.steprunner(batch)
            step_log = dict({self.stage + "_loss": loss}, **step_metrics)
            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


def train_model(net, optimizer, loss_fn, metrics_dict,
                train_data, val_data=None,
                epochs=10, ckpt_path='checkpoint.pt',
                patience=5, monitor="val_loss", mode="min"):
    history = {}

    for epoch in range(1, epochs + 1):
        printlog("Epoch {0} / {1}".format(epoch, epochs))

        # 1，train -------------------------------------------------
        train_step_runner = StepRunner(net=net, stage="train",
                                       loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict),
                                       optimizer=optimizer)
        train_epoch_runner = EpochRunner(train_step_runner)
        train_metrics = train_epoch_runner(train_data)

        for name, metric in train_metrics.items():
            history[name] = history.get(name, []) + [metric]

        # 2，validate -------------------------------------------------
        if val_data:
            val_step_runner = StepRunner(net=net, stage="val",
                                         loss_fn=loss_fn, metrics_dict=deepcopy(metrics_dict))
            val_epoch_runner = EpochRunner(val_step_runner)
            with torch.no_grad():
                val_metrics = val_epoch_runner(val_data)
            val_metrics["epoch"] = epoch
            for name, metric in val_metrics.items():
                history[name] = history.get(name, []) + [metric]

        # 3，early-stopping -------------------------------------------------
        arr_scores = history[monitor]
        best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
        print("<<<<<< this epoch {0} : {1} >>>>>>".format(monitor, arr_scores[len(arr_scores) - 1]))
        if best_score_idx == len(arr_scores) - 1:
            # torch.save(net.state_dict(), ckpt_path)
            print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                              arr_scores[best_score_idx]))
        # if len(arr_scores) - best_score_idx > patience:
        #     print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
        #         monitor, patience))
        #     break
        # net.load_state_dict(torch.load(ckpt_path))
        torch.save(net.state_dict(), ckpt_path)
        visualization(net, "figure", "../data", ".cache")

    return pd.DataFrame(history)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    net = net().to('cuda' if torch.cuda.is_available() else 'cpu')
    net.load_state_dict(torch.load('checkpoint.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = loss().step1
    metrics_dict = {"ssim": StructuralSimilarityIndexMeasure().to('cuda' if torch.cuda.is_available() else 'cpu'),
                    "lpips": LearnedPerceptualImagePatchSimilarity('squeeze', normalize=False).to(
                        'cuda' if torch.cuda.is_available() else 'cpu'),
                    "psnr": PeakSignalNoiseRatio().to('cuda' if torch.cuda.is_available() else 'cpu')}
    data_loader = DataLoader(dataset('../data/LIMHC', '.cache'), batch_size=25, shuffle=True, num_workers=32)
    train_model(net, optimizer, loss_fn, metrics_dict, data_loader, monitor="train_loss", epochs=1000)
