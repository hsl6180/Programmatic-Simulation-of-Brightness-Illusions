# 图像亮度恒常性与边缘感知项目

## 项目概述
针对机器测量灰度值相同但人眼感知亮度不同的情况，编写代码模拟这种视觉现象。
![image](https://github.com/user-attachments/assets/72ef9257-f80a-471d-8873-fda0f149f737)

## 实验效果展示
### 亮度恒常性
![image](https://github.com/user-attachments/assets/7601c625-c9ed-40e1-87b1-0db4da4bb638)

### 边缘感知
![image](https://github.com/user-attachments/assets/50aa3e67-1501-4947-bcbb-a5081fcd85bc)


## 总体实验步骤
1. **选取图像**：从自然风景图像库中精心挑选 34 张图像，并将其统一缩放到 256*256 像素大小，确保图像尺寸的一致性，以便后续处理和训练。
2. **添加噪声**：对每张选定的图像依次叠加多种类型噪声，包括高斯模糊（sigma = 25）、运动模糊（角度 = 45，运动长度 = 20）、高斯噪声（mean = 0, sigma = 25）、椒盐噪声（salt_prob = 0.05, pepper_prob = 0.05）、泊松噪声、斑点噪声（mean = 0, sigma = 0.1）以及图像二值化（像素值归一化到 0 - 1，>0.5 归为 1，其余归为 0），模拟真实场景中的复杂噪声环境，增强模型的鲁棒性。
3. **Unet 去噪训练**：构建包含训练（Image 存放加噪声图像，Label 存放原始图像）和推理（Test 存放检测图像）部分的训练流程，基于特定的训练路径和 Unet 结构参数（Unet.py 中定义）进行 75 轮训练，使模型学习噪声图像到原始图像的映射关系，实现去噪和特征恢复。

## 具体实验步骤
### 选取图像
从丰富的自然风景图像资源中筛选出 34 张具有代表性的图像，利用图像处理工具将其尺寸标准化为 256*256 像素，为后续实验奠定基础。

### 添加噪声
针对每张图像，按照既定的噪声参数设置，运用专业的图像处理库进行噪声叠加操作。高斯模糊用于模拟光线扩散造成的模糊效果；运动模糊模拟物体运动产生的拖影；高斯噪声、椒盐噪声、泊松噪声和斑点噪声分别从不同概率分布和强度层面引入干扰；图像二值化则创造出简单的二值对比场景，全方位考验模型应对噪声的能力。

### Unet 训练
在训练阶段，严格按照规定的目录结构组织数据，确保 Image 目录下的噪声图像与 Label 目录下的原始图像一一对应，为模型提供准确的训练样本对。在训练过程中，依据 Unet.py 中精心设计的网络结构参数进行模型搭建，并通过 75 轮迭代训练不断优化模型权重，使其在处理各类噪声图像时能够准确恢复原始图像特征。

## 代码文件使用指南
项目代码结构清晰，核心代码文件 train.py 位于与 sample_data 同级目录下。在运行训练程序前，请确保 sample_data 目录中的 image、label 和 test 子目录已正确填充相应图像数据，且数据格式符合程序要求。Unet.py 文件定义了模型的详细架构，可根据实际需求和硬件条件对其中的参数进行调整，如网络层数、卷积核大小等，以平衡模型的性能和训练效率。
