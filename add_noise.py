import cv2
import numpy as np
import os
from scipy.ndimage import convolve
from skimage import io, img_as_float


# 封装高斯模糊操作
def apply_gaussian_blur(image, gaussian_std):
    return cv2.GaussianBlur(image, (0, 0), gaussian_std)


def motion_blur_filter(length, angle):
    """生成运动模糊滤波器。

    参数:
    - length: 运动模糊的长度。
    - angle: 运动模糊的角度。

    返回:
    - H: 运动模糊滤波器。
    """
    # 创建一个长度为 (length, length) 的零数组
    H = np.zeros((length, length), dtype=np.float32)

    # 将角度转换为弧度
    theta = np.deg2rad(angle)

    # 计算运动模糊的线
    # 计算线的起点和终点
    for i in range(length):
        x = int((length - 1) / 2 + i * np.cos(theta))
        y = int((length - 1) / 2 - i * np.sin(theta))

        if 0 <= x < length and 0 <= y < length:
            H[x, y] = 1
            
    # 对滤波器归一化
    H /= np.sum(H)

    return H

def apply_motion_blur(image, length, angle):
    """应用运动模糊到给定图像。

    参数:
    - image: 输入图像。
    - length: 运动模糊的长度。
    - angle: 运动模糊的角度。

    返回:
    - filtered_image: 应用运动模糊后的图像。
    """
    # 生成运动模糊滤波器
    H = motion_blur_filter(length, angle)

    # 使用cv2.filter2D应用卷积
    filtered_image = cv2.filter2D(image, -1, H)

    return filtered_image


# 封装添加高斯噪声操作
def add_gaussian_noise(image, mean, sigma):
    """向图像添加高斯噪声。

    参数:
    - image: 输入图像（应为灰度图像）。
    - mean: 噪声的均值，默认为0。
    - sigma: 噪声的标准差，默认为25。

    返回:
    - noisy_image: 添加高斯噪声后的图像。
    """
    # 生成与图像相同形状的高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    
    # 将噪声添加到图像
    noisy_image = image + gaussian_noise
    
    # 确保像素值在合法范围内
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)


# 封装添加椒盐噪声操作
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    """Add salt and pepper noise to an image."""
    noisy_image = np.copy(image)
    # Randomly select a fraction of pixels to add salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 1  # Salt noise (white)

    # Randomly select a fraction of pixels to add pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0  # Pepper noise (black)

    return noisy_image


# 封装添加泊松噪声操作（注意泊松噪声实现与实际可能有一定差异，只是简单模拟效果）
def add_poisson_noise(image):
    image_float = image.astype(np.float32)

    # 生成泊松噪声
    noisy_image = np.random.poisson(image_float)

    # 确保像素值在合法范围内
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)


# 封装添加斑点噪声操作（简单模拟，与标准实现有差异）
def add_speckle_noise(image, mean, std):
    """向图像添加斑点噪声。

    参数:
    - image: 输入图像（应为非负值的图像）。

    返回:
    - noisy_image: 添加斑点噪声后的图像。
    """
    # 确保输入图像为浮点类型
    image_float = image.astype(np.float32)

    # 生成与输入图像相同形状的高斯噪声
    noise = np.random.normal(mean, std, image_float.shape)  # 均值为0，标准差为0.1

    # 将噪声乘以原始图像（斑点噪声的特性）
    noisy_image = image_float + image_float * noise

    # 确保像素值在合法范围内
    noisy_image = np.clip(noisy_image, 0, 255)

    return noisy_image.astype(np.uint8)


# 设置源图片所在文件夹路径（从这里读取图片）
source_folder = "sample_data/label"
# 设置目标图片保存文件夹路径
destination_folder = "sample_data/image"
# 如果目标文件夹不存在，则创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 获取源文件夹下的所有图片文件列表（假设都是PNG格式，可根据实际情况修改扩展名判断逻辑）
image_files = [f for f in os.listdir(source_folder) if f.endswith('.jpg')]

# 是否可视化处理后的图像，True表示可视化，False表示不可视化，可根据需要调整
visualize_images = False

for img_name in image_files:
    # 构建输入图像文件名，从源文件夹读取图片
    im = os.path.join(source_folder, img_name)
    # 构建输出图像文件名，保存到目标文件夹
    wm = os.path.join(destination_folder, img_name)
    # 读取图像,转为灰度图
    I = cv2.imread(im, cv2.IMREAD_GRAYSCALE)

    if I is None:
        print(f"无法读取图片 {im}，可能文件不存在或者格式不正确，跳过此图片处理。")
        continue

    # 依次应用各种图像处理操作
    I = apply_gaussian_blur(I, 10)
    I = apply_motion_blur(I, 20, 45)
    I = add_gaussian_noise(I, 0, 25)
    I = add_salt_pepper_noise(I, 0.05, 0.05)
    I = add_poisson_noise(I)
    I = add_speckle_noise(I, 0, 0.1)

    if visualize_images:
        cv2.imshow("Image", I)
        cv2.waitKey(0)

    # 保存处理后的图像
    cv2.imwrite(wm, I)