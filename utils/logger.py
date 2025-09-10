import datetime, time
import os
import numpy as np
import torch
import torchvision.utils as vutils
from . import utils
from scipy.io import savemat


class Logger(object):
    def __init__(self, args):
        self.times = {'init': time.time()}
        self._checkPath(args)
        self.args = args
        self.printArgs()

    def printArgs(self):
        strs = '------------ Options -------------\n'
        strs += '{}'.format(utils.dictToString(vars(self.args)))
        strs += '-------------- End ----------------\n'
        # print(strs)

    def _checkPath(self, args):
        if hasattr(args, 'run_model') and args.run_model:
            log_root = os.path.join(os.path.dirname(args.retrain), 'run_model')
            utils.makeFiles([os.path.join(log_root, 'test')])
        else:
            if args.resume and os.path.isfile(args.resume):
                log_root = os.path.join(os.path.dirname(os.path.dirname(args.resume)), 'resume')
            else:
                log_root = os.path.join(args.save_root, args.item)
            for sub_dir in ['train', 'val']:
                utils.makeFiles([os.path.join(log_root, sub_dir)])
            args.cp_dir = os.path.join(log_root, 'train')
        args.log_dir = log_root

    def getTimeInfo(self, epoch, iters, batch):
        time_elapsed = (time.time() - self.times['init']) / 3600.0
        total_iters = (self.args.epochs - self.args.start_epoch + 1) * batch
        cur_iters = (epoch - self.args.start_epoch) * batch + iters
        time_total = time_elapsed * (float(total_iters) / cur_iters)
        return time_elapsed, time_total

    def printItersSummary(self, opt):
        epoch, iters, batch = opt['epoch'], opt['iters'], opt['batch']
        strs = ' | {}'.format(str.upper(opt['split']))
        strs += ' Iter [{}/{}] Epoch [{}/{}]'.format(iters, batch, epoch, self.args.epochs)
        if opt['split'] == 'train':
            time_elapsed, time_total = self.getTimeInfo(epoch, iters, batch)
            strs += ' Clock [{:.2f}h/{:.2f}h]'.format(time_elapsed, time_total)
            strs += ' LR [{}]'.format(opt['recorder'].records[opt['split']]['lr'][epoch][0])
        print(strs)
        if 'timer' in opt.keys():
            print(opt['timer'].timeToString())
        if 'recorder' in opt.keys():
            print(opt['recorder'].iterRecToString(opt['split'], epoch))

    def printEpochSummary(self, opt):
        split = opt['split']
        epoch = opt['epoch']
        print('---------- {} Epoch {} Summary -----------'.format(str.upper(split), epoch))
        print(opt['recorder'].epochRecToString(split, epoch))

    def saveNormalResults(self, results, split, epoch, iters, others=[], dataset_name='DiLiGenT_main'):
        if len(others) == 0:  # 针对Blobby、Sculpture数据集
            save_dir = os.path.join(self.args.log_dir, split)
            save_name = '%d_%d.png' % (epoch, iters)
            vutils.save_image(results, os.path.join(save_dir, save_name))
            results = results * 2.0 - 1.0
            Pred_Norm_mat = ((results.cpu().numpy())[0]).transpose(1, 2, 0)
            save_mat_name = '%d_%d.mat' % (epoch, iters)
            savemat(os.path.join(save_dir, save_mat_name), {'Normal_est': Pred_Norm_mat})
        else:  # 对其他数据集，要记录下来对应的名字，名字被放在others[0]中
            if (dataset_name == 'DiLiGenT_main') or (dataset_name == 'Sphere_Bunny_main'):
                save_dir = os.path.join(self.args.log_dir, split)
                save_name = others[0] + '.png'
                vutils.save_image(results, os.path.join(save_dir, save_name))
                results = results * 2.0 - 1.0
                Pred_Norm_mat = ((results.cpu().numpy())[0]).transpose(1, 2, 0)
                save_mat_name = others[0] + '.mat'
                savemat(os.path.join(save_dir, save_mat_name), {'Normal_est': Pred_Norm_mat})
            else:
                save_dir = os.path.join(self.args.log_dir, split)
                # save_name = others[0] + '.png'
                # vutils.save_image(results, os.path.join(save_dir, save_name))
                results = results * 2.0 - 1.0
                Pred_Norm_mat = ((results.cpu().numpy())[0]).transpose(1, 2, 0)
                save_mat_name = others[0] + '.mat'
                savemat(os.path.join(save_dir, save_mat_name), {'Normal_est': Pred_Norm_mat})
