import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import torch.nn.functional as F
import numpy as np
# import pandas as pd
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    # ����forward����
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_channels + growth_rate * i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()

        # ��ʼ�ľ��Ϊfilter:2����growth_rate
        num_init_feature = 2 * growth_rate

        # ��ʾcifar-10
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=1,
                                    padding=3, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        # self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))

        # self.classifier = nn.Linear(num_feature, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # features = self.features(x)
        # out = features.view(features.size(0), -1)
        # out = self.classifier(out)
        out = self.features(x)
        return out


# DenseNet_BC for ImageNet
def DenseNet121():
    # return DenseNet_BC(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000)
    return DenseNet_BC(growth_rate=32, block_config=(1, 2, 4, 3), num_classes=1000)


def DenseNet169():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=1000)


def DenseNet201():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=1000)


def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=1000, )


# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet_BC(growth_rate=12, block_config=(16, 16, 16))


class FeatExtractor1(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor1, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        # print('x:',x.shape)
        # out = self.conv1(x)
        #  print('1:',out.shape)
        # out = self.conv2(out)
        # print('2:',out.shape)
        # out_feat = self.conv3(out)
        # print('3:',out_feat.shape)
        # n, c, h, w = out_feat.data.shape
        #  out_feat   = out_feat.view(-1)
        out = self.conv1(x)
        # print('1:',out.shape)
        out = self.conv2(out)
        # print('2:',out.shape)
        out_feat = self.conv3(out)
        #  print('3:',out_feat.shape)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class FeatExtractor2(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(FeatExtractor2, self).__init__()
        self.other = other
        self.conv4 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv4(x)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.deconv(128, 64)
        self.deconv3 = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.deconv4 = DenseNet121()
        self.deconv5 = model_utils.deconv(188, 64)
        self.deconv6 = model_utils.conv(batchNorm, 64, 64, k=3, stride=2, pad=1)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class MyMFPSN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(MyMFPSN, self).__init__()
        self.extractor1 = FeatExtractor1(batchNorm, c_in, other)
        self.extractor2 = FeatExtractor2(batchNorm, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:  # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            # m =nn.Upsample(scale_factor=4, mode='nearest')
            # m(net_in)
            feat, shape = self.extractor1(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)

        featss = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            # m(net_in)
            feat, shape = self.extractor1(net_in)
            feat = feat.view(shape[0], shape[1], shape[2], shape[3])
            feat_fused = feat_fused.view(shape[0], shape[1], shape[2], shape[3])
            featt = torch.cat((feat, feat_fused), 1)
            featt, shapee = self.extractor2(featt)
            featss.append(featt)
        if self.fuse_type == 'mean':
            feat_fusedd = torch.stack(featss, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fusedd, _ = torch.stack(featss, 1).max(1)
        Normal_1 = self.regressor(feat_fusedd, shapee)

        L2Normals, L2_Albedo, Img_Max_Value, Img_Light_Max_Value, Img_Mean_Value, NonShadow, img_split, light_split = self.DataPreprocessing(
            x)

        return NonShadow, L2Normals, Normal_1




