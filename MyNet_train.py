import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import pandas as pd
from torchvision import transforms
from path_manager_new import PATH
from path_manager_new import VALPATH
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import random

INFO = PATH.info
VALINFO = VALPATH.info
LABEL = [0, 1]


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        voxel, label= sample

        return {voxel: torch.from_numpy(voxel),
                label: label}


class Md3vDataset(Dataset):

    def __init__(self, cropsize=32, transform=None):

        index = []
        index += list(INFO.index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = INFO['lable']
        self.transform = transform
        self.cropsize = cropsize

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        npz_temp = np.load(os.path.join(PATH.nodule_path, '%s.npz' % name))
        voxel_temp = npz_temp['voxel']
        voxel_temp = voxel_temp[50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2]
        voxel = voxel_temp[np.newaxis, ...]
        label = self.label[item]
        return voxel, label


class Md3vvalDataset(Dataset):

    def __init__(self, cropsize=32, transform=None):

        index = []
        index += list(VALINFO.index)
        self.index = tuple(sorted(index))  # the index in the info
        self.label = VALINFO['lable']
        self.transform = transform
        self.cropsize = cropsize

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        name = VALINFO.loc[self.index[item], 'name']
        npz_temp = np.load(os.path.join(VALPATH.nodule_path, '%s.npz' % name))
        voxel_temp = npz_temp['voxel']
        voxel_temp = voxel_temp[50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2]
        voxel = voxel_temp[np.newaxis, ...]
        label = self.label[item]
        return voxel, label


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):    # bn_size need to be 4
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=16, block_config=(4, 4, 4),
                 num_init_features=32, bn_size=4, drop_rate=0.5, num_classes=2):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=3, stride=1, padding=0, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool3d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.softmax(out, dim=1)
        return out


def main():

    transformed_dataset = Md3vDataset(cropsize=32, transform=ToTensor())
    transformed_valdataset = Md3vvalDataset(cropsize=32, transform=ToTensor())
    net = DenseNet()
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)

    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    batch_size = 32
    epoch_size = 1000
    for epoch in range(epoch_size):

        running_loss = 0
        net.train()
        for i, data in enumerate(DataLoader(transformed_dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=1), 0):
            npz_inputs, labels = data
            inputs = npz_inputs.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.2, torch.cuda.is_available())
            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs.data, labels)
            # loss = criterion(outputs, labels)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss / (i+1)
        print('epoch %d / %d loss: %.4f  ' % (epoch + 1, epoch_size, running_loss), end="")
        torch.save(net, './save/DenseNet_newv10_%d.pth' % epoch)

        correct = 0
        total = 0
        train_loss =0
        net.eval()
        for i, data in enumerate(DataLoader(transformed_dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=1), 0):
            npz_inputs, labels = data
            inputs = npz_inputs.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = criterion(outputs, labels)
            train_loss += loss.item()
        print('acc_train: %.2f %%  ' % (100 * float(correct) / total), end="")
        train_loss = train_loss / (i + 1)
        print('train_loss: %.4f  ' % train_loss, end="")

        correct = 0
        total = 0
        val_loss = 0
        for i, data in enumerate(DataLoader(transformed_valdataset, batch_size=batch_size,
                                                shuffle=False, num_workers=1), 0):
            npz_inputs, labels = data
            inputs = npz_inputs.type(torch.FloatTensor)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
        print('acc_val: %.2f %%  ' % (100 * float(correct) / total), end="")
        val_loss = val_loss / (i + 1)
        print('val_loss: %.4f' % val_loss)

    print('Finished Training')
    # net = torch.load('DenseNet.pkl')


if __name__ == '__main__':
    main()
