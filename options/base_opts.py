import argparse


class BaseOpts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):
        #### Device Arguments ####
        self.parser.add_argument('--cuda', default=True, action='store_false')
        self.parser.add_argument('--time_sync', default=False, action='store_true')
        self.parser.add_argument('--workers', default=2, type=int)
        self.parser.add_argument('--seed', default=0, type=int)
        self.parser.add_argument('--device', default='cuda:1')

        #### Model Arguments ####
        self.parser.add_argument('--fuse_type', default='max')
        self.parser.add_argument('--normalize', default=True, action='store_true')
        # self.parser.add_argument('--in_light', default=True, action='store_false') # 针对PS-FCN时，in_light参数为true
        self.parser.add_argument('--in_light', default=True, action='store_false')  # 针对UPS-FCN时，in_light参数为False
        self.parser.add_argument('--use_BN', default=False, action='store_true')  # 用BN效果不好，这里默认为False
        self.parser.add_argument('--train_img_num', default=32, type=int)  # for data normalization
        # self.parser.add_argument('--in_img_num', default=96, type=int)  # 针对DiLiGent数据集，这个参数不动
        self.parser.add_argument('--in_img_num', default=100, type=int)  # 针对DiLiGent102、DiLiGentPi、B&S数据集，这个参数不动
        # self.parser.add_argument('--in_img_num', default=64, type=int)
        self.parser.add_argument('--start_epoch', default=1, type=int)
        self.parser.add_argument('--epochs', default=30, type=int)
        self.parser.add_argument('--resume', default=None)
        # self.parser.add_argument('--retrain', default=None)
        self.parser.add_argument('--retrain',
                                 default='输入保存的位置')
        self.parser.add_argument('--save_root', default='data/Training/')
        self.parser.add_argument('--item', default='calib')

    def parse(self):
        self.args = self.parser.parse_args()
        return self.args
