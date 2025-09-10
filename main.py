import torch
from options import train_opts
from utils import logger, recorders
from datasets import custom_data_loader
from models import custom_model, solver_utils, model_utils
import os
import test_utils_diligent
from options import run_model_opts
import train_utils
import test_utils

'''Blobby、Sculpture上参数'''
args = train_opts.TrainOpts().parse()
log = logger.Logger(args)
args.retrain = None
'''DiliGent上参数'''
args_diligent = run_model_opts.RunModelOpts().parse()
args_diligent.model = args.model
log_diligent = logger.Logger(args_diligent)
'''模型训练用这个脚本文件，同时包含了在DiliGent数据集上的测试结果，同时尽量不保存Blobby、Sculpture的预测结果'''
'''因为之前训练中，loss值在0.2左右就不继续下降，分析是因为训练集和验证集被污染，这里结合每一个像素的遮挡信息进行分析'''


def main(args):
    '''数据集'''
    train_loader, val_loader = custom_data_loader.customDataloader(args)
    test_loader = custom_data_loader.benchmarkLoader(args_diligent)
    '''模型'''
    model = custom_model.buildModel(args)
    '''优化器等'''
    optimizer, scheduler, records = solver_utils.configOptimizer(args, model)
    criterion = solver_utils.Criterion(args)
    recorder = recorders.Records(args.log_dir, records)
    '''Blobby、Sculpture训练集上训练'''
    for epoch in range(args.start_epoch, args.epochs + 1):
        scheduler.step()
        recorder.insertRecord('train', 'lr', epoch, scheduler.get_lr()[0])
        train_utils.train(args, train_loader, model, criterion, optimizer, log, epoch, recorder)
        if epoch % args.val_intv == 0:
            model_utils.saveCheckpoint(args.cp_dir, epoch, model, optimizer, recorder.records, args)
    print('-' * 50)
    print('\n' * 5)
    print('-' * 50)
    '''Blobby、Sculpture验证集上实验结果'''
    model.load_state_dict(torch.load(os.path.join(args.cp_dir, 'checkp_best.pth.tar'))['state_dict'])
    test_utils.test(args, 'val', val_loader, model, log, 0, recorder)

    '''DiliGent数据集上实验结果'''
    recorder_diligent = recorders.Records(args_diligent.log_dir)
    test_utils_diligent.test_DiliGent(args_diligent, 'test', test_loader, model, log_diligent, 1, recorder_diligent)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    main(args)

