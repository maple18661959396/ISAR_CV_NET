import os
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from cv_isarnet.activation.ComplexActivation import complex_relu, complex_softmax
from cv_isarnet.layer.ComplexLayers import ComplexLinear, ComplexConv2d
from cv_isarnet.layer.ComplexDropout import ComplexDropout2D
from cv_isarnet.loss.ComplexLoss import ComplexAverageCrossEntropy, ComplexAverageCrossEntropyAbs
from cv_isarnet.layer.ComplexPooling import ComplexMaxPool2D
from cv_isarnet.function.ComplexBatchNorm import ComplexBatchNorm2d, ComplexBatchNorm1d
from cv_isarnet.layer.ComplexUpSampling import ComplexUpSamplingBilinear2d


def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []

        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.mat'):
                    self.file_list.append(os.path.join(root, file))
                    self.labels.append(os.path.basename(root))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        if label == "2S1":
            label = torch.tensor(0)
        elif label == "BRDM_2":
            label = torch.tensor(1)
        elif label == "D7":
            label = torch.tensor(2)
        elif label == "T62":
            label = torch.tensor(3)
        elif label == "ZIL131":
            label = torch.tensor(4)
        elif label == "ZSU_23_4":
            label = torch.tensor(5)
        else:
            label = torch.tensor(6)
        data = scipy.io.loadmat(file_path)['ComplexVal']
        numpy_complex_data = data.astype(np.complex64)
        torch_complex_data = torch.tensor(numpy_complex_data)
        target_height, target_width = 200, 200
        cutting = ComplexUpSamplingBilinear2d(size=(target_height, target_width))
        # Upsample the data to the size of 200*200
        data = cutting(torch_complex_data.unsqueeze(0).unsqueeze(0))
        # data.unsqueeze(0).unsqueeze(0): The data tensor is first unsqueeze(0) twice，
        # print(data.squeeze(0).shape)
        return data.squeeze(0), label


train_data_dir = r'E:\办公\研究生\方向与任务\新建文件夹\库函数上线\data\SOC\TRAIN'
test_data_dir = r'E:\办公\研究生\方向与任务\新建文件夹\库函数上线\data\SOC\TEST'

train_dataset = MyDataset(train_data_dir)
test_dataset = MyDataset(test_data_dir)


batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 16, 13, 1)
        self.bn2d1 = ComplexBatchNorm2d(16, track_running_stats=False)
        self.maxpool1 = ComplexMaxPool2D(2, 2)
        self.conv2 = ComplexConv2d(16, 32, 13, 1)
        self.bn2d2 = ComplexBatchNorm2d(32, track_running_stats=False)
        self.maxpool2 = ComplexMaxPool2D(2, 2)
        self.conv3 = ComplexConv2d(32, 64, 12, 1)
        self.bn2d3 = ComplexBatchNorm2d(64, track_running_stats=False)
        self.maxpool3 = ComplexMaxPool2D(2, 2)
        self.dropout1 = ComplexDropout2D(p=0.5)
        self.conv4 = ComplexConv2d(64, 128, 10, 1)
        self.bn2d4 = ComplexBatchNorm2d(128, track_running_stats=False)
        self.conv5 = ComplexConv2d(128, 7, 6, 1)
        self.bn2d5 = ComplexBatchNorm2d(7, track_running_stats=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2d1(x)
        x = complex_relu(x)
        # print(x.shape)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2d2(x)
        x = complex_relu(x)
        x = self.maxpool2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.bn2d3(x)
        x = complex_relu(x)
        x = self.maxpool3(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = self.bn2d4(x)
        x = complex_relu(x)
        x = self.conv5(x)
        x = self.bn2d5(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        x = complex_softmax(x, 1)
        # print(x.shape)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        # print(data.dtype)
        optimizer.zero_grad()
        output = model(data)

        # loss_function = ComplexAverageCrossEntropy()
        # Loss function use ComplexAverageCrossEntropy. Correspond to the following

        loss_function = ComplexAverageCrossEntropyAbs()
        # print(output.shape)
        # print(target.shape)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train\t Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )
            torch.save(model.state_dict(), './model.pth')
            torch.save(optimizer.state_dict(), './optimizer.pth')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    conf_matrix = torch.zeros(7, 7)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            loss_function = ComplexAverageCrossEntropy()
            test_loss += loss_function(output, target).item()

            # With ComplexAverageCrossEntropy loss function
            # correct_r = pred_1.eq(target.data.view_as(pred_1))
            # correct_i = pred_2.eq(target.data.view_as(pred_2))
            # print(correct_r)
            # print(correct_i)
            # for i in range(0, correct_r.shape[0]):
            #     if correct_r[i, 0] and correct_i[i, 0]:
            #         correct = correct+1
            # Real part and virtual part respectively and label verification
            # only both are correctly increased accuracy
            # pred_r = output.real
            # pred_i = output.imag
            # conf_matrix_1 = confusion_matrix(pred_r, target, conf_matrix)
            # conf_matrix_2 = confusion_matrix(pred_i, target, conf_matrix)
            # Real and imaginary parts make confusion matrices, respectively.

            output_abs = torch.abs(output)
            pred_3 = output_abs.data.max(1, keepdim=True)[1]
            correct += pred_3.eq(target.data.view_as(pred_3)).sum()
            conf_matrix_1 = confusion_matrix(output_abs, target, conf_matrix)

    test_loss = test_loss * 16 / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print(conf_matrix_1)
    # print(conf_matrix_2)


# Run training on 13 epochs
for epoch in range(1, 14):
    train(model, device, train_dataloader, optimizer, epoch)
    test(model, device, test_dataloader)
