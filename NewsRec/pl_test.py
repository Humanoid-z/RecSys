import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import torch.distributed as dist
import os

# 初始化进程组
def init_process(rank, size, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=size)
    print('init_process')
    fn(rank, size)

# 模型定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)

def train(rank, size):
    # 分发数据集
    print('train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=2, sampler=train_sampler)

    # 初始化模型和优化器
    model = Net().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    for epoch in range(5):  # 仅进行5个epoch的训练作为示例
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(rank), data[1].to(rank)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    # 每100个小批量打印一次损失
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 获取可用GPU数量
    rank = 0
    init_process(rank, world_size, train)
