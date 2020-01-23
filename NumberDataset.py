import torch
from torch.utils.data import Dataset


class NumberDataset(Dataset):#data.Dataset
    def __init__(self,length=10000):
        self.length = length
        self.data =(torch.rand([length,1])*length).int()
        self.data = self.data.float()
        self.label = self.data%2 # 0 means even number; 1 means odd number
        self.label = self.label.float()

    def __getitem__(self, index):
        data, target = self.data[index], self.label[index]
        return data, target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.length
