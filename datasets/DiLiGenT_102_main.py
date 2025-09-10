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
'''在DiLiGent102数据集上，给出类似DiLiGent数据集的输出结果'''
'''该数据集的图像为8位图像'''
Resize_Width, Resize_Height = 512, 512  # 在论文中要体现，说明性能不好，那是因为图像分辨率低


class DiLiGenT_102_main(data.Dataset):
    def __init__(self, args, split='train'):
        # self.root = os.path.join(args.bm_dir)
        self.root = '输入DiLiGent_102数据集位置'
        self.split = split
        self.args = args
        self.objs = ['BALL_ABS', 'BALL_ACRYLIC', 'BALL_Al', 'BALL_BAKELITE', 'BALL_Cu', 'BALL_NYLON', 'BALL_POM',
                     'BALL_PP', 'BALL_PVC', 'BALL_STEEL', 'BUNNY_ABS', 'BUNNY_ACRYLIC', 'BUNNY_Al', 'BUNNY_BAKELITE',
                     'BUNNY_Cu', 'BUNNY_NYLON', 'BUNNY_POM', 'BUNNY_PP', 'BUNNY_PVC', 'BUNNY_STEEL', 'GOLF_ABS',
                     'GOLF_ACRYLIC', 'GOLF_Al', 'GOLF_BAKELITE', 'GOLF_Cu', 'GOLF_NYLON', 'GOLF_POM', 'GOLF_PP',
                     'GOLF_PVC', 'GOLF_STEEL', 'HEXAGON_ABS', 'HEXAGON_ACRYLIC', 'HEXAGON_Al', 'HEXAGON_BAKELITE',
                     'HEXAGON_Cu', 'HEXAGON_NYLON', 'HEXAGON_POM', 'HEXAGON_PP', 'HEXAGON_PVC', 'HEXAGON_STEEL',
                     'NUT_ABS', 'NUT_ACRYLIC', 'NUT_Al', 'NUT_BAKELITE', 'NUT_Cu', 'NUT_NYLON', 'NUT_POM', 'NUT_PP',
                     'NUT_PVC', 'NUT_STEEL', 'PENTAGON_ABS', 'PENTAGON_ACRYLIC', 'PENTAGON_Al', 'PENTAGON_BAKELITE',
                     'PENTAGON_Cu', 'PENTAGON_NYLON', 'PENTAGON_POM', 'PENTAGON_PP', 'PENTAGON_PVC', 'PENTAGON_STEEL',
                     'PROPELLER_ABS', 'PROPELLER_ACRYLIC', 'PROPELLER_Al', 'PROPELLER_BAKELITE', 'PROPELLER_Cu',
                     'PROPELLER_NYLON', 'PROPELLER_POM', 'PROPELLER_PP', 'PROPELLER_PVC', 'PROPELLER_STEEL',
                     'SPIKE_ABS', 'SPIKE_ACRYLIC', 'SPIKE_Al', 'SPIKE_BAKELITE', 'SPIKE_Cu', 'SPIKE_NYLON', 'SPIKE_POM',
                     'SPIKE_PP', 'SPIKE_PVC', 'SPIKE_STEEL', 'SQUARE_ABS', 'SQUARE_ACRYLIC', 'SQUARE_Al',
                     'SQUARE_BAKELITE', 'SQUARE_Cu', 'SQUARE_NYLON', 'SQUARE_POM', 'SQUARE_PP', 'SQUARE_PVC',
                     'SQUARE_STEEL', 'TURBINE_ABS', 'TURBINE_ACRYLIC', 'TURBINE_Al', 'TURBINE_BAKELITE', 'TURBINE_Cu',
                     'TURBINE_NYLON', 'TURBINE_POM', 'TURBINE_PP', 'TURBINE_PVC', 'TURBINE_STEEL']
        self.names = util.readList(os.path.join(self.root, 'BALL_ABS', 'filenames.txt'), sort=False)
        self.l_dir, self.intens = self.get_light_direction_intensity()
        print('[%s Data] \t%d objs %d lights. Root: %s' %
              (split, len(self.objs), len(self.names), self.root))

    def get_light_direction_intensity(self):
        # 获取不同材料和形状下的光源方向、强度信息
        l_dir, l_intens = {}, {}
        for obj in self.objs:
            obj_direction_txt_path = os.path.join(self.root, obj, 'light_directions.txt')
            l_dir[obj] = np.genfromtxt(obj_direction_txt_path)
            obj_intensity_txt_path = os.path.join(self.root, obj, 'light_intensities.txt')
            l_intens[obj] = np.genfromtxt(obj_intensity_txt_path)
        return l_dir, l_intens

    def _getMask(self, obj):
        mask = imread(os.path.join(self.root, obj, 'mask.png'))
        if mask.ndim > 2: mask = mask[:, :, 0]
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        mask = cv.resize(mask, (Resize_Width, Resize_Height), interpolation=cv.INTER_LINEAR)

        mask = mask[:, :, np.newaxis]
        return mask / 255.0

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, index):
        np.random.seed(index)
        obj = self.objs[index]
        select_idx = np.random.permutation(len(self.names))[:len(self.names)]  # 测试时读取所有图像
        img_list = [os.path.join(self.root, obj, self.names[i]) for i in select_idx]
        intens = [np.diag(1 / self.intens[obj][i]) for i in select_idx]

        # GT_Normal = np.rand(Resize_Width, Resize_Height, 3)  # 随机生成，因为训练中遇不到，只为了保证代码的连贯性
        GT_Normal = 0.1 * np.random.random(size=(Resize_Width, Resize_Height, 3))  # 随机生成，因为预测中遇不到，只为了保证代码的连贯性
        imgs = []
        for idx, img_name in enumerate(img_list):
            # img = imread(img_name).astype(np.float32) / 255.0
            img = cv.cvtColor(cv.imread(img_name), cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img = cv.resize(img, (Resize_Width, Resize_Height), interpolation=cv.INTER_LINEAR)
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
            GT_Normal = np.pad(GT_Normal, ((0, pad_h), (0, pad_w), (0, 0)), 'constant',
                               constant_values=((0, 0), (0, 0), (0, 0)))

        img = img * mask.repeat(img.shape[2], 2)
        item = {'N': GT_Normal, 'img': img, 'mask': mask}
        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(self.l_dir[obj][select_idx]).view(-1, 1, 1).float()
        item['obj'] = obj
        return item
