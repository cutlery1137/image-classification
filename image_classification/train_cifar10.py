# -*- coding:utf-8 -*-
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from model.LeNet import LeNet
from model.AlexNet import AlexNet
from model.ResNet import resnet18

transform_lenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_alexnet = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
transform_resnet = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

train_data = torchvision.datasets.CIFAR10("./dataset/CIFAR10_dataset", train=True, transform=transform_lenet, download=True)
test_data = torchvision.datasets.CIFAR10("./dataset/CIFAR10_dataset", train=False, transform=transform_lenet, download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

logs_path = "train_logs/lenet"
writer = SummaryWriter(logs_path)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LeNet()
# model = AlexNet(num_classes=10, init_weights=True)

# ResNet不使用迁移学习
# model = resnet18(num_classes=10)

# ResNet使用迁移学习
# model = resnet18()
# model.load_state_dict(torch.load("weight/resnet18-5c106cde.pth"))
# inchannel = model.fc.in_features
# outchannel = 10 # num_classes
# model.fc = nn.Linear(inchannel, outchannel)

model.to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 30
train_step = 0

for epoch in range(epochs):
    print("--------第{}轮训练开始---------".format(epoch + 1))
    model.train()
    for data in train_dataloader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        if(train_step % 100 == 0):
            print('第{}次训练的损失为{}'.format(train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), train_step)

    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        accuracy = 0
        for data in test_dataloader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            accuracy += (outputs.argmax(1) == targets).sum()

        print('测试集上的总损失为{}'.format(total_test_loss))
        print('测试集上的正确率为{}'.format(accuracy / test_data_size))
        writer.add_scalar('test_loss', total_test_loss, epoch)
        writer.add_scalar('test_accuracy', accuracy / test_data_size, epoch)

torch.save(model.state_dict(), "lenet_save.pth")
writer.close()