from __future__ import division
import os
import numpy as np
from imageio import imread
import scipy.io as sio
import cv2 as cv
import torch
import torch.utils.data as data

from datasets import pms_transforms
from . import util

np.random.seed(0)


class DiLiGenT_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.root = os.path.join(
            './data/DiLiGenT/pmsData/')
        self.split = split
        self.args = args
        self.objs = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG', 'gobletPNG', 'harvestPNG', 'pot1PNG',
                     'pot2PNG', 'readingPNG']
        self.names = util.readList(os.path.join(self.root, 'ballPNG', 'filenames.txt'), sort=False)
        print('[%s Data] \t%d objs %d lights. Root: %s' %
              (split, len(self.objs), len(self.names), self.root))

        self.l_dir, self.intens = self.get_light_direction_intensity()
        # self.l_dir = util.light_source_directions()
        # self.intens = {}
        # intens_name = 'light_intensities.txt'
        # print('Files for intensity: %s' % (intens_name))
        # for obj in self.objs:
        #     self.intens[obj] = np.genfromtxt(os.path.join(self.root, obj, intens_name))

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:, :, 0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        return mask / 255.0

    def get_light_direction_intensity(self):
        # 获取不同材料和形状下的光源方向、强度信息
        l_dir, l_intens = {}, {}
        for obj in self.objs:
            obj_direction_txt_path = os.path.join(self.root, obj, 'light_directions.txt')
            l_dir[obj] = np.genfromtxt(obj_direction_txt_path)
            obj_intensity_txt_path = os.path.join(self.root, obj, 'light_intensities.txt')
            l_intens[obj] = np.genfromtxt(obj_intensity_txt_path)
        return l_dir, l_intens

    def __getitem__(self, index):
        np.random.seed(index)
        obj = self.objs[index]

        select_idx = np.random.permutation(len(self.names))[:self.args.in_img_num]

        img_list = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        intens = [np.diag(1 / self.intens[obj][i]) for i in select_idx]

        normal_path = os.path.join(self.root, obj, 'Normal_gt.mat')
        normal = sio.loadmat(normal_path)
        normal = normal['Normal_gt']

        imgs = []
        for idx, img_name in enumerate(img_list):
            img = imread(img_name).astype(np.float32) / 255.0
            # img = cv.cvtColor(cv.imread(img_name, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB) / 65535.0
            img = np.dot(img, intens[idx])
            imgs.append(img)
        if self.args.normalize:
            imgs = pms_transforms.normalize(imgs)
        img = np.concatenate(imgs, 2)
        if self.args.normalize:
            img = img * np.sqrt(len(imgs) / self.args.train_img_num)  # TODO

        mask = self._getMask(obj)
        down = 4
        if mask.shape[0] % down != 0 or mask.shape[1] % down != 0:
            pad_h = down - mask.shape[0] % down
            pad_w = down - mask.shape[1] % down
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w), (0, 0)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            normal = np.pad(normal, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                            constant_values=((0, 0), (0, 0), (0, 0)))
        img = img * mask.repeat(img.shape[2], 2)
        '''为了方便模型的运行，对DiLiGent数据集进行裁剪，变成512*512大小，与DiLiGent_100、DiLiGent_Pi尺度保持一致'''
        normal = normal[:, 50:50 + 512, :]
        img = img[:, 50:50 + 512, :]
        mask = mask[:, 50:50 + 512, :]

        item = {'N': normal, 'img': img, 'mask': mask}

        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(self.l_dir[obj][select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        return item

    def __len__(self):
        return len(self.objs)
