import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms
from cv_isarnet.activation.ComplexActivation import complex_relu, complex_softmax
from cv_isarnet.layer.ComplexLayers import ComplexFlatten, ComplexLinear
from cv_isarnet.loss.ComplexLoss import ComplexAverageCrossEntropy


batch_size = 64
n_train = 5000
n_test = 100
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
train_set = Subset(train_set, torch.arange(n_train))
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)
test_set = Subset(test_set, torch.arange(n_test))

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        self.fl = ComplexFlatten(start_dim=1)
        self.fc1 = ComplexLinear(784, 256)
        self.fc2 = ComplexLinear(256, 10)

    def forward(self, x):
        x = self.fl(x)
        x = self.fc1(x)
        x = complex_relu(x)
        x = self.fc2(x)
        x = complex_softmax(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss_function = ComplexAverageCrossEntropy()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).type(torch.complex64), target.to(device)
            output = model(data)
            loss_function = ComplexAverageCrossEntropy()
            test_loss += loss_function(output, target).item()
            pred_1 = output.real.data.max(1, keepdim=True)[1]
            pred_2 = output.imag.data.max(1, keepdim=True)[1]
            correct += (pred_1.eq(target.data.view_as(pred_1)).sum() + pred_2.eq(target.data.view_as(pred_2)).sum())/2
    test_loss = test_loss * 64 / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Run training on 7 epochs
for epoch in range(1, 8):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
