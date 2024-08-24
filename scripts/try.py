import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义条件扩散模型
class ConditionalDiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(ConditionalDiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim + num_classes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        y_onehot = torch.zeros(y.size(0), 10).to(x.device).scatter_(1, y.view(-1, 1), 1)
        x = torch.cat([x, y_onehot], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义扩散过程
def diffusion_process(x, t, y, model):
    noise = torch.randn_like(x)
    return x + t * noise + model(x, y)

# 定义损失函数
def loss_fn(x, x_recon):
    return torch.mean((x - x_recon) ** 2)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据集
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化模型
input_dim = 784  # MNIST图像大小为28x28
hidden_dim = 256
output_dim = 784
num_classes = 10
model = ConditionalDiffusionModel(input_dim, hidden_dim, output_dim, num_classes)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 150
for epoch in range(num_epochs):
    for data in dataloader:
        images, labels = data
        images = images.view(-1, input_dim)
        
        # 扩散过程
        t = torch.rand(1)  # 随机时间步
        x_recon = diffusion_process(images, t, labels, model)
        
        # 计算损失
        loss = loss_fn(images, x_recon)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成")

# 生成条件图像并去噪
def generate_and_denoise_image(model, input_dim, label):
    # 生成随机噪声图像
    noise_image = torch.randn(1, input_dim)
    
    # 去噪过程
    t = torch.tensor([0.5])  # 选择一个时间步
    denoised_image = diffusion_process(noise_image, t, torch.tensor([label]), model)
    
    # 反归一化
    denoised_image = denoised_image.view(28, 28)
    denoised_image = (denoised_image * 0.5) + 0.5
    
    return denoised_image

# 生成并可视化去噪后的图像
for i in range(0, 9):
    label = i  # 选择一个标签
    denoised_image = generate_and_denoise_image(model, input_dim, label)
    plt.imshow(denoised_image.detach().numpy(), cmap='gray')
    plt.title(f"Denoised Image for Label {label}")
    plt.savefig(f"/mnt/nas/share2/home/lwh/tryimg/{label}.png")