import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from collections import OrderedDict

'B&S数据集上MAE=6.182，DiliGent数据集上MAE=6.714，已经保存'


class FeatureExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=9, other={}):
        super(FeatureExtractor, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(c_in, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True), )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True), )

    def forward(self, x):
        feature_4 = self.conv_1(x)
        feature_2 = self.conv_2(feature_4)
        return feature_4, feature_2


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True), )
        self.conv = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1), )

    def forward(self, feat_4, feat_2):
        feat = self.conv_4(feat_4) + feat_2
        normal = F.normalize(self.conv(feat), p=2, dim=1)
        return normal


class MyMethod(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=6, other={}):
        super(MyMethod, self).__init__()
        self.extractor = FeatureExtractor(batchNorm=batchNorm, c_in=c_in, other=other)
        self.regressor = Regressor(batchNorm=batchNorm, other=other)

    def DataPreprocessing(self, x):
        with torch.no_grad():
            img, light = x
            Batchsize, NC, H, W = light.shape
            N = NC // 3
            ImgReshape, LightReshape = torch.reshape(img, (Batchsize, N, 3, H, W)), torch.reshape(light,
                                                                                                  (Batchsize, N, 3,
                                                                                                   H, W))[:, :, :, 0, 0]
            Img_gray = 0.2989 * ImgReshape[:, :, 0, :, :] + 0.5870 * ImgReshape[:, :, 1, :, :] + 0.1140 * ImgReshape[:,
                                                                                                          :, 2, :, :]
            '''利用L2的方法求解一下表面法向量、表面反射率'''
            TransferConv = nn.Conv2d(in_channels=N, out_channels=3, kernel_size=1, stride=1, padding=0,
                                     bias=False).to(torch.device(img.device))
            L2Normals = []
            for i in range(Batchsize):
                TransferConv.weight.data = torch.linalg.pinv(LightReshape[i]).unsqueeze(2).unsqueeze(3)
                L2Normals.append(TransferConv(Img_gray[i].unsqueeze(0)))
            L2Normals = torch.cat(L2Normals, dim=0)
            L2Normals = F.normalize(L2Normals, p=2, dim=1)
            LightReshape = torch.reshape(light, (Batchsize, N, 3, H, W))
            ReLU = nn.ReLU()
            L2_Albedo = (torch.sum(ImgReshape, 1)) / (
                    torch.sum(ReLU(torch.sum(L2Normals.unsqueeze(1) * LightReshape, dim=2)).unsqueeze(2),
                              dim=1) + 1e-6)
            '''利用像素亮度最大值的方式，找到能让像素点最亮时的光照方向'''
            Img_gray_reshape = Img_gray.unsqueeze(2)
            _, Img_gray_Max_Indice = torch.max(Img_gray_reshape, dim=1, keepdim=True)
            Img_Light_Max_Value = torch.gather(torch.reshape(light, (Batchsize, N, 3, H, W)), dim=1,
                                               index=Img_gray_Max_Indice.expand(-1, -1, 3, -1, -1))
            Img_Light_Max_Value = Img_Light_Max_Value.squeeze(1)
            Img_Max_Value, _ = torch.max(ImgReshape, dim=1, keepdim=True)
            Img_Max_Value = Img_Max_Value.squeeze(1)
            '''计算输入图像信息的平均亮度'''
            Img_Mean_Value = torch.mean(ImgReshape, dim=1, keepdim=True).squeeze(1)
            '''获取每一组的输入信息'''
            img_split, light_split = torch.split(img, 3, 1), torch.split(light, 3, 1)  # 每一个都是列表，包含有N个[B, 3, H, W]数据
            NonShadow = []
            for i in range(len(img_split)):
                img_gray_i = 0.2989 * img_split[i][:, 0, :, :] + 0.5870 * img_split[i][:, 1, :, :] + 0.1140 * \
                             img_split[i][:, 2, :, :]
                img_gray_i = img_gray_i.unsqueeze(1)
                NonShadow.append(torch.tensor(img_gray_i > 0.05, dtype=torch.float32))  # 像素值太低，可能意味着有遮挡
            NonShadow = torch.sum(torch.stack(NonShadow, dim=1), dim=1) / N
            NonShadow = (NonShadow > 0.25)

            return L2Normals, L2_Albedo, Img_Max_Value, Img_Light_Max_Value, Img_Mean_Value, NonShadow, img_split, light_split

    def forward(self, x):
        L2Normals, L2_Albedo, Img_Max_Value, Img_Light_Max_Value, Img_Mean_Value, NonShadow, img_split, light_split = self.DataPreprocessing(
            x)
        for i in range(len(img_split)):
            feat_i_4, feat_i_2 = self.extractor(torch.cat([img_split[i], light_split[i]], dim=1))
            if i == 0:
                feat_4, feat_2 = feat_i_4, feat_i_2
            else:
                feat_4, _ = torch.stack([feat_4, feat_i_4], 1).max(1)
                feat_2, _ = torch.stack([feat_2, feat_i_2], 1).max(1)
        Normal_1 = self.regressor(feat_4, feat_2)
        return NonShadow, L2Normals, Normal_1


if __name__ == '__main__':
    device = torch.device('cuda:0')
    B, N, C, H, W = 4, 32, 3, 32, 32
    rand_x = [torch.rand(size=(B, N * C, H, W)).to(device), torch.rand(size=(B, N * C, H, W)).to(device)]
    model = MyMethod().to(device)
    NonShadow, Normal_1, Normal_2 = model(rand_x)
    print(Normal_1.shape, torch.max(Normal_1), torch.min(Normal_1), torch.linalg.norm(Normal_1[0, :, 0, 0]))
    print(Normal_2.shape, torch.max(Normal_2), torch.min(Normal_2), torch.linalg.norm(Normal_2[0, :, 0, 0]))
