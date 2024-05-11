## 图像分类
参考链接：

b站同济子豪兄：[Pytorch迁移学习训练自己的图像分类模型【两天搞定AI毕设】](https://www.bilibili.com/video/BV1Ng411C7WY/?spm_id_from=333.788&vd_source=230eaadd335ba8a7ecff8d53d3abf218)

b站霹雳吧啦Wz：[6.2 使用pytorch搭建ResNet并基于迁移学习训练](https://www.bilibili.com/video/BV14E411H7Uw/?spm_id_from=333.788&vd_source=230eaadd335ba8a7ecff8d53d3abf218)



train_logs文件夹：存放tensorboard日志文件

```py
tensorboard --logdir=train_logs
```

weight文件夹：存放ResNet18预训练权重

### LeNet训练MNIST

MNIST是28*28的灰度图，LeNet的输入的图像是32\*32的，要修改LeNet的第一个卷积层（即conv1）：

```py
model = LeNet()
model.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
```

灰度图的通道为1，因此in_channels=1；共有6个卷积核，out_channels=6；padding=2可以让28*28的图像变成32\*32

注意：图像预处理的Normalize()不要写成三通道！

```py
transform_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])
```

类似地，在MNIST上训练AlexNet和ResNet18，需要将第一个卷积层的in_channels改为1

### LeNet训练cifar10

图像预处理：

```python
transform_ = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

权重初始化：使用了kaiming初始化，也可以换成xavier，可以对比下效果。

权重初始化的两种方式：

1. 权重初始化的函数放在模型定义中声明，在 \__init__() 函数中使用

```py
class LeNet(nn.Module):
    def __init__(self):
		# 定义各层
        self._initialize_weights() # 使用权重初始化

    def forward(self, x):
		# 定义前向传播
        return x
    
	# 声明权重初始化函数
    def _initialize_weights(self): 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
```

2. 直接在训练中声明和使用（参考李沐《动手学深度学习》LeNet部分）

```py
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        
net.apply(init_weights)
```

### AlexNet训练cifar10

权重初始化：kaiming初始化适合ReLU，xavier初始化适合softmax和tan

```py
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
```
图像预处理：

```python
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
```

AlexNet使用了dropout，我们不希望在验证集中使用dropout，因此要加上model.train和model.eval()

### ResNet18训练cifar10（使用迁移学习方式）

图像预处理：由于采用迁移学习的方式训练，Normalize采用固定值（ImageNet的均值和标准差）

```py
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}
```

注意：ResNet使用了BN层，要加上model.train()和model.eval()。

ResNet加载预训练权重有两种方式：

1. resnet18()不传参数，不修改线性层，先加载预训练权重，再把linear层的outchannel改为5（顺序反了会报错）

```py
model = resnet18()
model.load_state_dict(torch.load("weight/resnet18-5c106cde.pth"))
inchannel = model.fc.in_features
outchannel = 10 # num_classes
model.fc = nn.Linear(inchannel, outchannel)
```

2. 可以先将线性层的outchannel变为5，由于torch.load()导入进来的是字典，把字典中线性层的参数删掉，再load_state_dict

### 迁移学习

斯坦福CS231N【迁移学习】中文精讲：https://www.bilibili.com/video/BV1K7411W7So

斯坦福CS231N【迁移学习】官方笔记：https://cs231n.github.io/transfer-learning

ResNet算法精讲与论文逐句精读：https://www.bilibili.com/video/BV1vb4y1k7BV

#### 选择一：只微调训练模型最后一层（全连接分类层）


```
model = models.resnet18(pretrained=True) # 载入预训练模型
# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)
```

```
model.fc
```

```
Linear(in_features=512, out_features=30, bias=True)
```

```
# 只微调训练最后一层全连接层的参数，其它层冻结
optimizer = optim.Adam(model.fc.parameters())
```

#### 选择二：微调训练所有层

```
model = models.resnet18(pretrained=True) # 载入预训练模型
model.fc = nn.Linear(model.fc.in_features, n_class)
optimizer = optim.Adam(model.parameters())
```

#### 选择三：随机初始化模型全部权重，从头训练所有层

```
model = models.resnet18(pretrained=False) # 只载入模型结构，不载入预训练权重参数
model.fc = nn.Linear(model.fc.in_features, n_class)
optimizer = optim.Adam(model.parameters())
```

