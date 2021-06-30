import sys
sys.path.append('./')
sys.path.append('../')
sys.path.append('./example')
from easydict import EasyDict as edict
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from tqdm import tqdm
import pickle

from zfnet import ZFNet, fuse_zfnet
from qqquantize.quantize import prepare
from qqquantize.observers.histogramobserver import HistogramObserver
from qqquantize.observers.fake_quantize import FakeQuantize
from qqquantize.observers.fake_quantize import enable_fake_quant, disable_fake_quant, enable_observer, disable_observer
from qqquantize.savehook import register_intermediate_hooks

FLOAT_CKPT = './checkpoint/zfnet_float.pth'
CIFAR_ROOT = '/home/luojiapeng/root_data_lnk/datasets/cifar'
DEVICE = 'cuda'

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(testloader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print('test_acc: %.3f' % acc)

# Training
def train(net, trainloader, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        _lr = optimizer.param_groups[0]['lr']
    print('training Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.6f'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, _lr))

if __name__ == '__main__':
    net = ZFNet(0.5).eval().to(DEVICE)
    ckpt = torch.load(FLOAT_CKPT)
    net.load_state_dict(ckpt['net'])
    fuse_zfnet(net, inplace=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(
        root=CIFAR_ROOT, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4)
    print('>>> before quantize test')
    test(net, testloader)

    qconfig = edict({
        'activation': FakeQuantize.with_args(bits=8, max_factor=0.8),
        'weight': FakeQuantize.with_args(bits=8, max_factor=0.8),
        'bias': FakeQuantize.with_args(bits=8, max_factor=0.8)
    })

    net = prepare(net, qconfig).to(DEVICE)
    disable_fake_quant(net)
    enable_observer(net)
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        outputs = net(inputs.to(DEVICE))
    enable_fake_quant(net)
    disable_observer(net)
    print('>>> after quantize test')
    test(net, testloader)

    # qat
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01,
                          momentum=0.9, weight_decay=5e-4)
    def lr_lambda(epoch):
        if epoch < 50:
            return 1
        elif epoch < 100:
            return 0.2
        else:
            return 0.04
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for epoch in range(150):
        enable_observer(net)
        train(net, trainloader, criterion, optimizer)
        lr_scheduler.step()
        disable_observer(net)
        test(net, testloader)

    # output quantized intermediate
    hook = register_intermediate_hooks(net)
    loader_iter = iter(testloader)
    inputs, targets = next(loader_iter)
    net(inputs.to(DEVICE))
    inter_data = hook.output_data()
    pickle.dump(inter_data, open('inter_data.pkl', 'wb'))

    # save state_dict
    torch.save(net.state_dict(), 'checkpoint/zfnet_quant.pth')
