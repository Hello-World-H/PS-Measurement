import torch
import random
import numpy as np

# from skimage.transform import resize

random.seed(0)
np.random.seed(0)
import cv2 as cv


def arrayToTensor(array):
    if array is None:
        return array
    array = np.transpose(array, (2, 0, 1))
    tensor = torch.from_numpy(array)
    return tensor.float()


def rgbToGray(img):
    h, w, c = img.shape
    img = img[:, :, 0] * 0.229 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    return img.reshape(h, w, 1)


def normalToMask(normal, thres=0.01):
    """
    Due to the numerical precision of uint8, [0, 0, 0] will save as [127, 127, 127] in gt normal,
    When we load the data and rescale normal by N / 255 * 2 - 1, the [127, 127, 127] becomes 
    [-0.003927, -0.003927, -0.003927]
    """
    mask = (np.square(normal).sum(2, keepdims=True) > thres).astype(np.float32)
    return mask


def randomCrop(inputs, target, size):
    if not __debug__: print('RandomCrop: input, target', inputs.shape, target.shape, size)
    h, w, _ = inputs.shape
    c_h, c_w = size
    if h == c_h and w == c_w:
        return inputs, target
    x1 = random.randint(0, w - c_w)
    y1 = random.randint(0, h - c_h)
    inputs = inputs[y1: y1 + c_h, x1: x1 + c_w]
    target = target[y1: y1 + c_h, x1: x1 + c_w]
    del h, w, _, c_h, c_w, x1, y1
    return inputs, target


def rescale(inputs, target, size):
    if not __debug__: print('Rescale: Input, target', inputs.shape, target.shape, size)
    in_h, in_w, _ = inputs.shape
    h, w = size
    if h != in_h or w != in_w:
        # inputs = resize(inputs, size, order=1, mode='reflect')
        # target = resize(target, size, order=1, mode='reflect')
        inputs = cv.resize(inputs, (w, h), interpolation=cv.INTER_LINEAR)
        target = cv.resize(target, (w, h), interpolation=cv.INTER_LINEAR)
    del in_h, in_w, _, h, w
    return inputs, target


def randomNoiseAug(inputs, noise_level=0.05):
    if not __debug__: print('RandomNoiseAug: input, noise level', inputs.shape, noise_level)
    noise = np.random.random(inputs.shape)
    noise = (noise - 0.5) * noise_level
    inputs += noise
    inputs = np.clip(inputs, 0.0, 1.0)
    return inputs


def normalize(imgs):
    h, w, c = imgs[0].shape
    imgs = [img.reshape(-1, 1) for img in imgs]
    img = np.hstack(imgs)
    norm = np.sqrt((img * img).clip(0.0).sum(1))
    img = img / (norm.reshape(-1,1) + 1e-10)
    imgs = np.split(img, img.shape[1], axis=1)
    imgs = [img.reshape(h, w, -1) for img in imgs]
    return imgs


def normalize_gpt(imgs):
    # 获取图像的形状
    h, w, c = imgs.shape
    num_channels = c // 3
    # # 将图像重塑为 (h * w, num_channels, 3)
    img = imgs.reshape(h * w, num_channels, 3).transpose(0, 2, 1).reshape(-1, num_channels)
    # # 计算每个像素的范数
    norm = np.linalg.norm(img, axis=1, keepdims=True)
    # # 归一化图像
    img = img / (norm + 1e-10)
    # # 将归一化后的图像恢复为原始形状
    img = img.reshape(h, w, 3, num_channels).transpose(0, 1, 3, 2).reshape(h, w, -1)
    return img
