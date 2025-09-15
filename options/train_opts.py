from .base_opts import BaseOpts


class TrainOpts(BaseOpts):
    def __init__(self):
        super(TrainOpts, self).__init__()
        self.initialize()

    def initialize(self):
        BaseOpts.initialize(self)
        #### Dataset Arguments ####
        self.parser.add_argument('--dataset', default='PS_Synth_Dataset')
        self.parser.add_argument('--data_dir',
                                 default='./data/PS_Blobby_Dataset')
        self.parser.add_argument('--data_dir2',
                                 default='./data/PS_Sculpture_Dataset')
        self.parser.add_argument('--concat_data', default=True, action='store_true')  # 训练和测试都需要数据的拼接

        self.parser.add_argument('--train_rescale', default=False, action='store_false')
        self.parser.add_argument('--train_crop', default=True, action='store_false')
        self.parser.add_argument('--train_crop_h', default=64, type=int)
        self.parser.add_argument('--train_crop_w', default=64, type=int)
        self.parser.add_argument('--train_noise_aug', default=True, action='store_false')
        self.parser.add_argument('--train_noise', default=0.001, type=float)
        self.parser.add_argument('--train_color_aug', default=True, action='store_false')
        self.parser.add_argument('--train_in_img_num', default=32, action='store_false')
        '------------------------------------------------------------------------------'
        self.parser.add_argument('--test_rescale', default=False, action='store_false')
        self.parser.add_argument('--test_crop', default=False, action='store_false')
        self.parser.add_argument('--test_crop_h', default=128, type=int)
        self.parser.add_argument('--test_crop_w', default=128, type=int)
        self.parser.add_argument('--test_noise_aug', default=False, action='store_false')
        self.parser.add_argument('--test_noise', default=0.0, type=float)
        self.parser.add_argument('--test_color_aug', default=False, action='store_false')
        self.parser.add_argument('--test_in_img_num', default=64, action='store_false')

        #### Training Arguments ####
        self.parser.add_argument('--model', default='MyMethod')

        self.parser.add_argument('--solver', default='adam', help='adam|sgd')
        self.parser.add_argument('--milestones', default=[5, 10, 15, 20, 25], nargs='+', type=int)
        self.parser.add_argument('--init_lr', default=1e-3, type=float)
        self.parser.add_argument('--lr_decay', default=0.5, type=float)
        self.parser.add_argument('--beta_1', default=0.9, type=float, help='adam')
        self.parser.add_argument('--beta_2', default=0.999, type=float, help='adam')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='sgd')
        self.parser.add_argument('--batch', default=32, type=int)
        self.parser.add_argument('--val_batch', default=8, type=int)

        #### Display Arguments ####
        self.parser.add_argument('--train_disp', default=60, type=int)
        self.parser.add_argument('--train_save', default=200, type=int)
        self.parser.add_argument('--val_intv', default=1, type=int)
        self.parser.add_argument('--val_disp', default=1, type=int)
        self.parser.add_argument('--val_save', default=1, type=int)

        #### Checkpoint Arguments ####
        self.parser.add_argument('--save_intv', default=1, type=int)

        #### Loss Arguments ####
        self.parser.add_argument('--normal_loss', default='cos', help='cos|mse')
        self.parser.add_argument('--normal_w', default=1)

    def parse(self):
        BaseOpts.parse(self)
        self.args.train_img_num = self.args.in_img_num  # for data normalization
        return self.args
