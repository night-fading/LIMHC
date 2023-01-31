import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.nn import MSELoss

from datasets import trainDataset
from model.encoder import encoder


def train():
    loss = MSELoss()
    net = encoder().to('mps')
    optimizer = optim.Adam(net.parameters())
    net.train()
    data_loader = torch.utils.data.DataLoader(trainDataset('../data/LIMHC', '.cache'), batch_size=20, shuffle=True)
    for i in range(20):
        for data in data_loader:
            img_t, img_cropped_t, input_encoder, position = data
            img_t = img_t.to('mps')
            img_cropped_t = img_cropped_t.to('mps')
            input_encoder = input_encoder.to('mps')
            x = net(input_encoder)
            out = torch.sum(torch.abs(img_cropped_t - x)) / 20
            net.zero_grad()
            out.backward()
            optimizer.step()
            print(out.item())
    img_t, img_cropped_t, input_encoder, location = trainDataset('../data/LIMHC', '.cache').__getitem__(3)
    img_t = img_t.to('mps')
    plt.imshow(img_cropped_t.permute(1, 2, 0))
    plt.show()
    img_cropped_t = img_cropped_t.to('mps')
    input_encoder = input_encoder.to('mps')
    x = net(input_encoder.unsqueeze(0))
    plt.imshow(x.to('cpu').detach().squeeze(0).permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    train()
