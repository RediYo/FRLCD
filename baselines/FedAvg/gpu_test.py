import torch

# 检查是否有可用的 GPU 设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)


# 定义一个神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 创建一个神经网络模型并将其移动到 GPU 上
net = Net().to(device)

# 定义一个随机输入张量并将其移动到 GPU 上
inputs = torch.randn(3, 10).to(device)

# 在 GPU 上计算模型输出
outputs = net(inputs)

# 打印模型输出
print("模型输出：", outputs)
