import os
import numpy as np
import torch
from torch.utils.data import Dataset
from MyNet_train import _DenseBlock, _DenseLayer, _Transition, DenseNet, ToTensor
from test_path_manager import PathManager
import csv

TESTPATH = PathManager()
INFO = TESTPATH.info


class Md3vtestDataset(Dataset):
    def __init__(self, cropsize=32, transform=None):
        index = []
        index += list(INFO.index)
        self.index = tuple(sorted(index))  # the index in the info
        self.transform = transform
        self.cropsize = cropsize

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        name = INFO.loc[self.index[item], 'name']
        npz_temp = np.load(os.path.join(TESTPATH.nodule_path, '%s.npz' % name))
        voxel_temp = npz_temp['voxel']
        voxel_temp = voxel_temp[50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2,
                     50 - self.cropsize // 2:50 + self.cropsize // 2]
        voxel = voxel_temp[np.newaxis, ...]
        return name, voxel

def main():
    csvFile = open("final_submit.csv", "w", newline='')
    writer = csv.writer(csvFile)
    writer.writerow(["Id", "Predicted"])
    test_dataset = Md3vtestDataset(cropsize=32, transform=ToTensor())
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False, num_workers=1)
    net1 = torch.load('net1.pth', map_location='cpu')
    net1.eval()
    device = torch.device("cpu")
    net1.to(device)
    net2 = torch.load('net2.pth', map_location='cpu')
    net2.eval()
    device = torch.device("cpu")
    net2.to(device)
    with torch.no_grad():
        for data in testloader:
            name, voxel = data
            voxel = voxel.type(torch.FloatTensor)
            voxel = voxel.to(device)
            outputs1 = net1(voxel)
            outputs2 = net2(voxel)
            writer.writerow([str(name[0]), ((outputs1.data[0][1]).item()+(outputs2.data[0][1]).item())/2])
            print(str(name[0]), ((outputs1.data[0][1]).item()+(outputs2.data[0][1]).item())/2, 'writed')
    csvFile.close()


if __name__ == '__main__':
    main()

