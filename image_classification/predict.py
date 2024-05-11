import torch
import torchvision.transforms as transforms
from PIL import Image

from model.LeNet import LeNet

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_path = "./image/dog.png"
image = Image.open(image_path)
image = image.convert("RGB") #png格式是四通道，转换成三通道

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
image = transform(image)
print(image.shape) # torch.Size([3, 32, 32])
image = torch.reshape(image, (1, 3, 32, 32)) # 也可以 image = torch.unsqueeze(image, dim=0)

model = LeNet()
model.load_state_dict(torch.load("lenet_save.pth"))

model.eval()
with torch.no_grad():
    output = model(image)
    predict = torch.argmax(output, dim=1)

print(classes[int(predict)])