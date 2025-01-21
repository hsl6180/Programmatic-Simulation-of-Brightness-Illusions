import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt

# 假设这里有定义好的unet模型结构，示例中简单导入，实际需要根据具体定义来
from unet import NestedUNet, UNet

def histogram_equalization(img):
    # img 的范围为 0-1
    img_flat = img.flatten()  # 将图像展平为一维数组
    histogram, bins = np.histogram(img_flat, bins=256, range=[0, 1])  # 计算直方图

    cdf = histogram.cumsum()  # 计算累积分布函数
    cdf_normalized = cdf / cdf.max()  # 归一化到 0-1 范围
    
    # 应用直方图均衡化
    img_equalized = np.interp(img_flat, bins[:-1], cdf_normalized)  # 归一化
    return img_equalized.reshape(img.shape)  # 将展平的结果重新变回原图形状

def adjustData(img, mask):
    # 确保输入是 numpy 数组
    img = np.array(img)
    mask = np.array(mask)
    
    # 进行直方图归一化
    # img = histogram_equalization(img)
    
    # 将 mask 转换为二值图像 (0或1)
    mask_binary = (mask > 0.5).astype(np.float32)  # 将大于0的值设为1，其余设为0
    
    return img, mask_binary

# 自定义数据集类，用于训练数据
class TrainDataset(Dataset):
    def __init__(self, train_path, image_folder, mask_folder, target_size=(256, 256), flag_multi_class=False, num_class=2):
        self.train_path = train_path
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.target_size = target_size
        self.flag_multi_class = flag_multi_class
        self.num_class = num_class
        self.image_paths = self._get_image_paths()
        self.mask_paths = self._get_mask_paths()
        self.transform = self._get_transforms()

    def _get_image_paths(self):
        return sorted(glob.glob(os.path.join(self.train_path, self.image_folder, "*.jpg")))

    def _get_mask_paths(self):
        return sorted(glob.glob(os.path.join(self.train_path, self.mask_folder, "*.jpg")))

    def _get_transforms(self):
        transform_list = [
            transforms.Resize(self.target_size),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
            # transforms.RandomHorizontalFlip(),  # 随机水平翻转
            # transforms.RandomVerticalFlip(),  # 随机竖直翻转
            # transforms.RandomCrop(self.target_size, padding=4),  # 随机裁剪，带填充
            # transforms.RandomRotation(degrees=10),  # 随机旋转
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        img = self.transform(img)
        mask = self.transform(mask)

        img, mask = adjustData(img, mask)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return img, mask

def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"---Using device: {device}","---")
    resizeto = (256, 256)
    
    # 创建训练数据集和数据加载器
    train_dataset = TrainDataset('sample_data', 'image', 'label', resizeto)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = UNet(1,1).to(device)  # 将模型移动到GPU
    # model = NestedUNet(1, 1, deep_supervision=True).to(device)  # 将模型移动到GPU
    criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(75):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据和目标移动到GPU
            data, target = data.to(device), target.to(device)
            # imshow(torchvision.utils.make_grid(data.cpu(), normalize=True), "Data")
            # imshow(torchvision.utils.make_grid(target.cpu(), normalize=True), "Target")
            optimizer.zero_grad()

            # 一般模型
            output = model(data)
            loss = criterion(output, target)

            # # unet++的deep_supervision=True时开启
            # outputs = model(data)
            # loss = sum(criterion(output, target) for output in outputs)  # 计算所有输出的损失
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch: {epoch + 1}, Loss: {running_loss}')
    
    # 保存训练后的模型
    model_save_path = "trained_model_unet256_75.pth"  # 定义模型保存路径
    outpath = "./model/" + model_save_path
    torch.save(model.state_dict(), outpath)  # 保存模型权重
    print(f"Model saved to {model_save_path}")

    