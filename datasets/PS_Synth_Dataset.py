from __future__ import division
import os
import numpy as np
# from scipy.ndimage import imread
import cv2 as cv
# import time
import torch
import torch.utils.data as data
# import gc
from datasets import pms_transforms
from . import util

np.random.seed(0)
# cpu一个一个读取，每一次（包含__getitem__全过程）用时相差很多，有时0.1s，有时0.02s

class PS_Synth_Dataset(data.Dataset):
    def __init__(self, args, root, split='train'):
        self.root = os.path.join(root)
        self.split = split
        self.args = args
        self.shape_list = util.readList(os.path.join(self.root, split + '_mtrl.txt'), sort=False)
        if split == 'train':
            self.training_state = 'train'
        else:
            self.training_state = 'test'

    def _getInputPath(self, index):
        shape, mtrl = self.shape_list[index].split('/')
        root = self.root
        normal_path = root + '/' + 'Images/' + shape + '/' + shape + '_normal.png'
        img_dir = root + '/' + 'Images/' + self.shape_list[index]
        img_list = util.readList(img_dir + '/' + shape + '_' + mtrl + '.txt', sort=False)
        data = np.genfromtxt(img_list, dtype='str', delimiter=' ')
        if self.training_state == 'train':
            select_idx = np.random.permutation(data.shape[0])[:self.args.train_in_img_num]
        else:
            select_idx = np.random.permutation(data.shape[0])[:self.args.test_in_img_num]
        data = data[select_idx, :]
        imgs = list((np.char.add(np.array(img_dir + '/'), data[:, 0])).reshape(-1))
        lights = data[:, 1:4].astype(np.float32)
        del shape, mtrl, root, img_dir, img_list, data, select_idx,
        return normal_path, imgs, lights

    def __getitem__(self, index):
        if self.training_state == 'train':
            normal_path, img_list, lights = self._getInputPath(index)
            normal = cv.cvtColor(cv.imread(normal_path), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 2.0 - 1

            imgs_array = np.empty((128, 128, 3 * len(img_list)), dtype=np.uint8)
            # 读取和存储图像
            for idx, img_path in enumerate(img_list):
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
                imgs_array[:, :, idx * 3:(idx + 1) * 3] = img
            # 在所有图像读取完毕后，进行类型转换和归一化
            img = imgs_array.astype(np.float32) / 255.0

            h, w, c = img.shape
            crop_h, crop_w = self.args.train_crop_h, self.args.train_crop_w
            if self.args.train_rescale:
                sc_h = np.random.randint(crop_h, h)
                sc_w = np.random.randint(crop_w, w)
                img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

            if self.args.train_crop:
                img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])
            # #
            if self.args.train_color_aug and not self.args.normalize:
                img = (img * np.random.uniform(1, 3)).clip(0, 2)
            # #
            if self.args.normalize:
                img = pms_transforms.normalize_gpt(img)

            # #
            if self.args.train_noise_aug:
                img = pms_transforms.randomNoiseAug(img, self.args.train_noise)
            mask = pms_transforms.normalToMask(normal)
            normal = normal * mask.repeat(3, 2)
            norm = np.sqrt((normal * normal).sum(2, keepdims=True))
            normal = normal / (norm + 1e-10)

            #
            # # 对下面进行了修改，以提升训练速度
            normal = pms_transforms.arrayToTensor(normal)
            img = pms_transforms.arrayToTensor(img)
            mask = pms_transforms.arrayToTensor(mask)

            if self.args.in_light:
                lights = torch.from_numpy(lights).view(-1, 1, 1).float()
                # del h, w, c, sc_h, sc_w, img_list, idx, imgs_array, img_path, normal_path, crop_h, crop_w, norm
                return (normal, img, mask, lights)
            else:
                # del h, w, c, sc_h, sc_w, img_list, idx, imgs_array, img_path, normal_path, crop_h, crop_w, norm
                return (normal, img, mask)
        else:
            normal_path, img_list, lights = self._getInputPath(index)
            normal = cv.cvtColor(cv.imread(normal_path), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0 * 2.0 - 1

            imgs_array = np.empty((128, 128, 3 * len(img_list)), dtype=np.uint8)
            # 读取和存储图像
            for idx, img_path in enumerate(img_list):
                img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
                imgs_array[:, :, idx * 3:(idx + 1) * 3] = img
            # 在所有图像读取完毕后，进行类型转换和归一化
            img = imgs_array.astype(np.float32) / 255.0

            h, w, c = img.shape
            crop_h, crop_w = self.args.test_crop_h, self.args.test_crop_w
            if self.args.test_rescale:
                sc_h = np.random.randint(crop_h, h)
                sc_w = np.random.randint(crop_w, w)
                img, normal = pms_transforms.rescale(img, normal, [sc_h, sc_w])

            if self.args.test_crop:
                img, normal = pms_transforms.randomCrop(img, normal, [crop_h, crop_w])
            # #
            if self.args.test_color_aug and not self.args.normalize:
                img = (img * np.random.uniform(1, 3)).clip(0, 2)
            # #
            if self.args.normalize:
                img = pms_transforms.normalize_gpt(img)

            # #
            if self.args.test_noise_aug:
                img = pms_transforms.randomNoiseAug(img, self.args.test_noise)
            mask = pms_transforms.normalToMask(normal)
            normal = normal * mask.repeat(3, 2)
            norm = np.sqrt((normal * normal).sum(2, keepdims=True))
            normal = normal / (norm + 1e-10)

            #
            # # 对下面进行了修改，以提升训练速度
            normal = pms_transforms.arrayToTensor(normal)
            img = pms_transforms.arrayToTensor(img)
            mask = pms_transforms.arrayToTensor(mask)

            if self.args.in_light:
                lights = torch.from_numpy(lights).view(-1, 1, 1).float()
                # del h, w, c, img_list, idx, imgs_array, img_path, normal_path, crop_h, crop_w, norm
                return (normal, img, mask, lights)
            else:
                # del h, w, c, img_list, idx, imgs_array, img_path, normal_path, crop_h, crop_w, norm
                return (normal, img, mask)

    def __len__(self):
        return len(self.shape_list)
