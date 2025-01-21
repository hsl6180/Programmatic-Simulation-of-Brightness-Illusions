import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from unet import NestedUNet, UNet
from torchvision.utils import save_image

# 自定义数据集类，用于测试数据
class TestDataset(Dataset):
    def __init__(self, test_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
        self.test_path = test_path
        self.target_size = target_size
        self.flag_multi_class = flag_multi_class
        self.image_paths = self._get_image_paths()
        self.transform = self._get_transforms()

    def _get_image_paths(self):
        return sorted(glob.glob(os.path.join(self.test_path, "*.png")))

    def _get_transforms(self):
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor()
        ]
        return transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')
        img = self.transform(img)
        return img

def imshow(img, title):
    npimg = img.numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"---Using device: {device}","---")

resizeto = (256, 256)

# 创建测试数据集和数据加载器（这里batch_size设为1示例，根据实际调整）
test_dataset = TestDataset("sample_data/test", resizeto)
test_loader = DataLoader(test_dataset, batch_size=1)

model = UNet(1, 1).to(device)  # 将模型移动到GPU
# model = NestedUNet(1, 1, deep_supervision=True).to(device)
modelpath = "./model/" + "trained_model_unet256_75_best.pth"
model.load_state_dict(torch.load(modelpath))
model.eval() 
with torch.no_grad():
    for i, data in enumerate(test_loader):
        # 将测试数据移动到GPU
        data = data.to(device)
        # imshow(torchvision.utils.make_grid(data.cpu(), normalize=True), "Data")

        # 一般模型
        output = model(data)
        # imshow(torchvision.utils.make_grid(output.cpu(), normalize=True), "Data")
        

        # unet++的deep_supervision=True时开启
        # output = model(data)[0]
        # imshow(torchvision.utils.make_grid(outputs[0].cpu(), normalize=True), "Data")
        save_image(output.cpu(), os.path.join("./output", f'output_{i}.png'), normalize=True)