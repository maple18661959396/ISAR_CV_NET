import torch
import torch.nn as nn
import numpy as np
import os
class Net(nn.Module):
    def __int__(self):
        super(Net, self).__int__()
        self.layer1 = nn.Conv2d(1,10,5,1)
        self.layer2 = nn.Linear(10,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    net_test = Net()
    # 打印网络中所有类内变量的信息（按照先后顺序
    print(net_test)
    # 打印网络中所有类内变量的参数值
    print(net_test.state_dict())
    # 打印网络构成的参数字典中所有的网络键值，之后根据这个键值就可以去查看固定哪一层的参数
    # 然后通过索引甚至可以看到具体这一层的第几个参数
    print(net_test.state_dict().keys())
    # 通过字典键值索引打印某一个键值下面的参数
    print(net_test.state_dict()["layer2.bias"])
    print(net_test.state_dict()["layer2.bias"].shape)