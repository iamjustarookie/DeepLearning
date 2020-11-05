#使用letnet5实现 cifar100 GPU版
import torch
from lenet5.lenet import Lenet5
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn, optim
#from visdom import Visdom
#import numpy as np
#from lenet5.visual_loss import Visualizer
#from torchnet import meter

# 超参数

criteon = nn.CrossEntropyLoss()
model = Lenet5().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 256


# 获取数据
def getData():
    # 一张
    cifar_train = datasets.CIFAR10("cifar", train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),

    ]))

    # 多线程 下载所有

    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)

    # 一张
    cifar_test = datasets.CIFAR10("cifar", train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    total_num = len(cifar_test)
    # 多线程 下载所有
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)

    return cifar_train, cifar_test, total_num


# 迭代训练
def process(data_train, data_test, total_num):
    #vis = Visualizer(env='main')
    #loss_meter = meter.AverageValueMeter()
    for epoch in range(100):
        #loss_meter.reset()
        model.train()
        for step, (x, label) in enumerate(data_train):
            x=x.cuda()
            logits = model(x)
            label=label.cuda()
            logits=logits.cuda()
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #loss_meter.add(loss.item())
            #vis.plot_many_stack({'train_loss': loss_meter.value()[0]})  # 为了可视化增加的内容
            if step % 10 == 0:
                print("loss:", loss.item(),"epoch:",epoch)
        test(data_test, total_num)


# 检验 正确度
def test(data_test, total_num):
    total_correct = 0
    #vis = Visualizer(env='main')
   # acc_meter = meter.AverageValueMeter()
    #acc_meter.reset()
    model.eval()
    for step, (x, label) in enumerate(data_test):
        x=x.cuda()
        logits = model(x)
        logits=logits.cuda()
        pred = logits.argmax(dim=1)
        label=label.cuda()
        correct = pred.eq(label).sum().float().item()
        total_correct += correct
        #acc_meter.add(total_correct)
        #vis.plot_many_stack({'acc': acc_meter.value()[0]})  # 为了可视化增加的内容
    acc = total_correct / total_num
    print("正确个数：", total_correct)
    print("正确率：", acc)


# 主函数
def main():
    data_train, data_test, total_num = getData()
    process(data_train, data_test, total_num)


if __name__ == "__main__":
    main()
