from __future__ import print_function
import argparse
from math import log10
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ESPCN
from utils_model import get_training_set, get_test_set, set_dtype
import parser_model as parser

print('Note: This script cannot be executed without the train and test set, which \
        were not included in this repository.')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
CUDA = torch.cuda.is_available()
dtype = set_dtype(CUDA)
device = torch.device("cuda" if CUDA else "cpu")


train_set = get_training_set(args.upscale_factor)
test_set = get_test_set(args.upscale_factor)
train_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, \
                                batch_size=args.train_batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, \
                                batch_size=args.test_batch_size, shuffle=False)

net = ESPCN(upscale_factor=args.upscale_factor).to(device)

# Uncomment below to load trained weights
# weights = torch.load('data/weights/weights_epoch_30.pth')
# net.load_state_dict(weights)

criterion = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=args.lr)

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_data_loader, 1):
        img_in, target = batch[0].to(device).type(dtype), \
                         batch[1].to(device).type(dtype)

        optim.zero_grad()
        loss = criterion(net(img_in), target)
        epoch_loss += loss.item()

        loss.backward()
        optim.step()

    avg_loss = epoch_loss / len(train_data_loader)
    print("\n---> Epoch {} Avg. Loss: {:.4f}".format(epoch, avg_loss))
    return


def test():
    tot_mse = 0
    with torch.no_grad():
        for _, batch in enumerate(test_data_loader, 1):
            img_in, target = batch[0].to(device).type(dtype), \
                             batch[1].to(device).type(dtype)
            prediction = net(img_in)

            mse = criterion(prediction, target)
            tot_mse += mse

    avg_loss = tot_mse / len(test_data_loader)
    avg_psnr = 10 * log10(255.0**2/ avg_loss)
    print("---> Avg. PSNR: {:.4f} dB".format(avg_psnr))
    return


def checkpoint(epoch):
    weights_path = 'data/weights/weights_epoch_{}.pth'.format(epoch)
    torch.save(net.state_dict(),weights_path)



if __name__ == '__main__':

    for epoch in range(1, args.num_epochs + 1):
        train(epoch)
        test()
        checkpoint(epoch)