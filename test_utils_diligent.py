import os
import torch
import torchvision.utils as vutils
import numpy as np
from models import model_utils
from utils import eval_utils, time_utils

'''该代码针对diligent系列数据集，相对于Blobby、Sculpture数据集，该方法在保存图像时（.mat或者.png），能够体现出对应目标的名字'''


def get_itervals(args, split):
    args_var = vars(args)
    disp_intv = args_var[split + '_disp']
    save_intv = args_var[split + '_save']
    return disp_intv, save_intv


def test_DiliGent(args, split, loader, model, log, epoch, recorder):
    # 对于DiLiGent及其系列数据集，由于有地面真值（对于DiLiGent系列数据集，其GT随机给出，不具有参考性，只为了保证代码顺利运行），所以可以计算acc
    model.eval()
    print('---- Start %s Epoch %d: %d batches ----' % (split, epoch, len(loader)))
    timer = time_utils.Timer(args.time_sync)
    disp_intv, save_intv = get_itervals(args, split)
    MAE_List = []
    with torch.no_grad():
        for i, sample in enumerate(loader):
            data = model_utils.parseData_DiliGent(args, sample, timer, split)
            input = model_utils.getInput(args, data)

            print(data['obj'])
            out_var = model(input)
            out_var = out_var[-1]
            timer.updateTime('Forward')
            acc, error_map = eval_utils.calNormalAcc(data['tar'].data, out_var.data, data['m'].data)
            MAE_List.append(acc['n_err_mean'])
            recorder.updateIter(split, acc.keys(), acc.values())

            iters = i + 1
            if iters % disp_intv == 0:
                opt = {'split': split, 'epoch': epoch, 'iters': iters, 'batch': len(loader),
                       'timer': timer, 'recorder': recorder}
                log.printItersSummary(opt)

            if iters % save_intv == 0:
                pred = (out_var.data + 1) / 2
                masked_pred = pred * data['m'].data.expand_as(out_var.data)
                log.saveNormalResults(masked_pred, split, epoch, iters, others=data['obj'], dataset_name=args.benchmark)
                log.saveErrorMap(error_map, split, epoch, iters, others=data['obj'], dataset_name=args.benchmark)

            del i, sample, data, input, out_var, acc, iters, pred, masked_pred

    opt = {'split': split, 'epoch': epoch, 'recorder': recorder}
    log.printEpochSummary(opt)
    Error_Test_Epoch = np.mean(opt['recorder'].records[opt['split']]['n_err_mean'][opt['epoch']])
    MAE_List = np.around(np.array(MAE_List), decimals=2)
    print(list(MAE_List))
    return Error_Test_Epoch, MAE_List

