import torch
import os


class Criterion(object):
    def __init__(self, args):
        self.setupNormalCrit(args)

    def setupNormalCrit(self, args):
        print('=> Using {} for criterion normal'.format(args.normal_loss))
        self.normal_loss = args.normal_loss
        self.normal_w = args.normal_w

        self.n_crit_mse = torch.nn.MSELoss(reduction='none')
        self.n_crit_cos = torch.nn.CosineEmbeddingLoss(reduction='none')

        if args.cuda:
            device = torch.device(args.device)
            self.n_crit_mse.to(device)
            self.n_crit_cos.to(device)

    def forward(self, output, target):

        num = target.nelement() // target.shape[1]
        if not hasattr(self, 'flag') or num != self.flag.nelement():
            self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1))
        NonShadow, Normal_1, Normal_2 = output
        NonShadow = NonShadow.permute(0, 2, 3, 1).contiguous().view(-1)
        target = target.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        # Normal_1 = Normal_1.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        Normal_2 = Normal_2.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        # loss_target_1 = torch.sum(self.n_crit_mse(Normal_1, target), dim=1) + self.n_crit_cos(Normal_1, target,
        #                                                                                       self.flag)
        loss_target = torch.sum(self.n_crit_mse(Normal_2, target), dim=1) + self.n_crit_cos(Normal_2, target,
                                                                                              self.flag)
        # loss_target = 0.25 * loss_target_1 + 0.75 * loss_target_2
        self.loss = (loss_target * NonShadow).mean() * (len(NonShadow) / torch.sum(NonShadow) + 1e-6)
        out_loss = {'N_loss': self.loss.item()}
        return out_loss

    def backward(self):
        self.loss.backward()


def getOptimizer(args, params):
    print('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer


def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones, gamma=args.lr_decay,
                                                     last_epoch=args.start_epoch - 2)
    return scheduler


def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch


def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        print("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
    #     args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
