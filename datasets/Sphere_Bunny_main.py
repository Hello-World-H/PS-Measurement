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
'''在Sphere_Bunny数据集上，给出类似DiLiGent数据集的输出结果'''
'''缺乏公开数据集结果，在论文中最多作为消融实验与鲁棒性实验（在大论文中使用，期刊论文不打算采用）'''
'''提取的格式可能与DiliGent数据集等格式不符合，可能不能直接运行，需要改变格式及尺度大小，才能运行'''


class SphereBunny_main(data.Dataset):
    def __init__(self, args, split='train'):
        self.Category = ['bunny_s-0.55_x-000_y-180', 'sphere_s-0.80_x-000_y-000'][1]  # 需要哪个用哪个
        self.root = './data/PS_Test_Sphere_Bunny/'
        self.split = split
        self.args = args
        self.objs = ['alum-bronze', 'alumina-oxide', 'aluminium', 'aventurnine', 'beige-fabric', 'black-fabric',
                     'black-obsidian', 'black-oxidized-steel', 'black-phenolic', 'black-soft-plastic', 'blue-acrylic',
                     'blue-fabric', 'blue-metallic-paint', 'blue-metallic-paint2', 'blue-rubber', 'brass', 'cherry-235',
                     'chrome', 'chrome-steel', 'colonial-maple-223', 'color-changing-paint1', 'color-changing-paint2',
                     'color-changing-paint3', 'dark-blue-paint', 'dark-red-paint', 'dark-specular-fabric', 'delrin',
                     'fruitwood-241', 'gold-metallic-paint', 'gold-metallic-paint2', 'gold-metallic-paint3',
                     'gold-paint', 'gray-plastic', 'grease-covered-steel', 'green-acrylic', 'green-fabric',
                     'green-latex', 'green-metallic-paint', 'green-metallic-paint2', 'green-plastic', 'hematite',
                     'ipswich-pine-221', 'light-brown-fabric', 'light-red-paint', 'maroon-plastic', 'natural-209',
                     'neoprene-rubber', 'nickel', 'nylon', 'orange-paint', 'pearl-paint', 'pickled-oak-260',
                     'pink-fabric', 'pink-fabric2', 'pink-felt', 'pink-jasper', 'pink-plastic', 'polyethylene',
                     'polyurethane-foam', 'pure-rubber', 'purple-paint', 'pvc', 'red-fabric', 'red-fabric2',
                     'red-metallic-paint', 'red-phenolic', 'red-plastic', 'red-specular-plastic', 'silicon-nitrade',
                     'silver-metallic-paint', 'silver-metallic-paint2', 'silver-paint', 'special-walnut-224',
                     'specular-black-phenolic', 'specular-blue-phenolic', 'specular-green-phenolic',
                     'specular-maroon-phenolic', 'specular-orange-phenolic', 'specular-red-phenolic',
                     'specular-violet-phenolic', 'specular-white-phenolic', 'specular-yellow-phenolic', 'ss440',
                     'steel', 'teflon', 'tungsten-carbide', 'two-layer-gold', 'two-layer-silver', 'violet-acrylic',
                     'violet-rubber', 'white-acrylic', 'white-diffuse-bball', 'white-fabric', 'white-fabric2',
                     'white-marble', 'white-paint', 'yellow-matte-plastic', 'yellow-paint', 'yellow-phenolic',
                     'yellow-plastic']

    def __len__(self):
        return len(self.objs)

    def __getitem__(self, index):
        # 从.txt文件中读取光源信息、图像名字
        NamesList, LightDirections = [], []
        with open(
                self.root + 'Images/' + self.Category + '/' + self.objs[index] + '/' + self.Category + '_' + self.objs[
                    index] + '.txt') as File:
            for line in File.readlines():
                line = line.replace('\n', '')
                name, LightDirection = line.split('.png ')
                NamesList.append(name + '.png')
                LightDirection = LightDirection.split(' ')
                LightDirections.append([float(LightDirection[0]), float(LightDirection[1]), float(LightDirection[2])])
        # 读取所有图像，并将其变成合适的形式，同时还有光源信息
        Images = []
        for i in range(len(NamesList)):
            # Images.append(cv.cvtColor(
            #     cv.imread(self.root + 'Images/' + self.Category + '/' + self.objs[index] + '/' + NamesList[i]),
            #     cv.COLOR_BGR2RGB).astype(np.float32) / 255.0)

            Images.append(
                imread(self.root + 'Images/' + self.Category + '/' + self.objs[index] + '/' + NamesList[i]).astype(
                    np.float32) / 255.0)
            # Images.append(
            #     sio.loadmat(self.root + 'MAT/' + self.Category + '/' + self.objs[index] + '/' + NamesList[i])['Image'])
        Images = np.concatenate(Images, 2)
        Images[Images > 1.0] = 1.0  # 事实上确实有像素大于1，应该是过曝光了
        LightDirections = np.array(LightDirections)

        if self.args.normalize:
            Images = pms_transforms.normalize_gpt(Images)
            Images = Images * np.sqrt(self.args.in_img_num / self.args.train_img_num)  # TODO

        # 读取法向量信息
        Normal_GT = sio.loadmat(self.root + 'MAT/' + self.Category + '.mat')['Normal']
        Mask_GT = imread(self.root + 'Images/' + self.Category + '/Mask.png') / 255.0
        Mask_GT[Mask_GT < 0.5] = 0.0
        Mask_GT[Mask_GT >= 0.5] = 1.0
        Mask_GT = np.expand_dims(Mask_GT, axis=-1)

        item = {'N': Normal_GT, 'img': Images, 'mask': Mask_GT}

        for k in item.keys():
            item[k] = pms_transforms.arrayToTensor(item[k])

        if self.args.in_light:
            item['light'] = torch.from_numpy(LightDirections).view(-1, 1, 1).float()

        item['obj'] = self.objs[index]

        return item


if __name__ == "__main__":
    pass
