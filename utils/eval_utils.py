import torch
import math
   
def calNormalAcc(gt_n, pred_n, mask=None):
    """Tensor Dim: NxCxHxW"""
    dot_product = (gt_n * pred_n).sum(1).clamp(-1,1)
    error_map   = torch.acos(dot_product) # [-pi, pi]
    angular_map = error_map * 180.0 / math.pi
    angular_map = angular_map * mask.narrow(1, 0, 1).squeeze(1)

    valid = mask.narrow(1, 0, 1).sum()
    ang_valid   = angular_map[mask.narrow(1, 0, 1).squeeze(1).byte()]
    n_err_mean  = ang_valid.sum() / valid
    n_err_med = ang_valid.median()  # clean
    n_acc_15 = (ang_valid < 15.0).sum().float() / valid.float()
    n_acc_30 = (ang_valid < 30.0).sum().float() / valid.float()
    value = {'n_err_mean': n_err_mean.item(), 'n_err_med': n_err_med.item(),
            'n_acc_15': n_acc_15.item(), 'n_acc_30': n_acc_30.item(),}
    return value
