import torch
import torch.nn as nn

__all__ = ['odevityNet_1', 'odevityNet_2', 'odevityNet_3', 'odevityNet_4','odevityNet_5','odevityNet_6']


class SinLayer(nn.Module):

    def __init__(self):
        super(SinLayer, self).__init__()
        # self.a = nn.parameter.Parameter(torch.rand(1))
        # self.b = nn.parameter.Parameter(torch.rand(1))
        self.fc = nn.Linear(1,1) # y = ax+b


    def forward(self, x):
        out=self.fc(x)


        return (torch.sin(out)/2+0.5)


class CosLayer(nn.Module):

    def __init__(self):
        super(CosLayer, self).__init__()
        self.fc = nn.Linear(1,1) # y = ax+b

    def forward(self, x):
        out=self.fc(x)
        return torch.cos(out)


class SinCosLayer(nn.Module):

    def __init__(self):
        super(SinCosLayer, self).__init__()
        self.fc1 = nn.Linear(1,1) # y = ax+b
        self.fc2 = nn.Linear(1, 1)  # y = ax+b

    def forward(self, x):
        out=torch.sin(self.fc1(x))
        out = torch.cos(self.fc2(out))
        return out


class OdevityNet(nn.Module):
    def __init__(self,layers=[SinLayer],nums=[1]):
        super(OdevityNet, self).__init__()
        self.layer = self._make_layer(layers, nums)
        # self.fc = nn.Linear(1,1)
        # self.a = nn.Parameter(torch.rand(1))
        # self.b = nn.Parameter(torch.rand(1))

    def _make_layer(self, layers, nums):
        Net_layers = []

        for layer in layers:
            for num in nums:
                for _ in range(num):
                    Net_layers.append(layer())
        return nn.Sequential(*Net_layers)

    def forward(self, x):
        out = self.layer(x)
        # out[out>=0.5]=1
        # out[out < 0.5] = 0
        return out
        # return out

def odevityNet_1( **kwargs):
    model = OdevityNet([SinLayer], [1], **kwargs)
    return model


def odevityNet_2( **kwargs):
    model = OdevityNet([CosLayer], [2], **kwargs)
    return model


def odevityNet_3( **kwargs):
    model = OdevityNet([SinCosLayer], [2], **kwargs)
    return model


def odevityNet_4( **kwargs):
    model = OdevityNet([SinLayer,SinCosLayer,SinLayer,CosLayer], [2,3,1,2], **kwargs)
    return model



def odevityNet_5( **kwargs):
    model = OdevityNet([SinLayer,SinCosLayer,SinLayer,CosLayer], [8,8,8,8], **kwargs)
    return model



def odevityNet_6( **kwargs):
    model = nn.Sequential(
        nn.Linear(1,16),
        nn.Tanh(),
        nn.Linear(16, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )
    return model

def demo():
    net = odevityNet_4()
    y = net(torch.randn(2,1))
    print(y.size())
    print(y)

# demo()
