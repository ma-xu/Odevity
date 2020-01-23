import torch
import random
from torch.utils.data import Dataset, DataLoader
import odevity as odevity
import argparse
from NumberDataset import NumberDataset
from torch.autograd import Variable

random = torch.randn(32)
print(random)


train_data = (torch.rand((50))*10000).int()
train_label = train_data%2 # 0 means even number; 1 means odd number

dataloader = DataLoader(train_data,
                        batch_size=3)







val_data = (torch.rand((5000))*10000).int()
val_label = val_data%2 # 0 means even number; 1 means odd number


# model_names = sorted(name for name in odevity.__all__)
# print(model_names)

parser = argparse.ArgumentParser(description='PyTorch odevity Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='odevityNet_1')
                    # choices=model_names)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
args = parser.parse_args()






def main():
    global args
    # print(odevity.['odevityNet_1']())

    model = odevity.odevityNet_1()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=4e-5)

    trainloader = torch
    print(trainloader)


    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)




def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """Comes from pytorch demo"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()





