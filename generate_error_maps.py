import os
import numpy as np
import torch
import math
import matplotlib as mpl
from scipy.io import loadmat, savemat
import torchvision.utils as vutils

def colorMap(diff, thres=90):
    """Convert angular error to color map"""
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = mpl.cm.jet(diff_norm)[:, :, :3]
    diff_cm_tensor = torch.from_numpy(diff_cm).permute(2, 0, 1).unsqueeze(0).float()
    return diff_cm_tensor

def calNormalAcc(gt_n, pred_n, mask=None):
    """Calculate normal accuracy and generate error map
    Args:
        gt_n: ground truth normal (H, W, 3)
        pred_n: predicted normal (H, W, 3)
        mask: mask (H, W)
    """
    # Convert to torch tensors if needed
    if not isinstance(gt_n, torch.Tensor):
        gt_n = torch.from_numpy(gt_n).float()
    if not isinstance(pred_n, torch.Tensor):
        pred_n = torch.from_numpy(pred_n).float()
    if mask is not None and not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).float()
    
    # Ensure shape is (H, W, 3)
    if gt_n.dim() == 3 and gt_n.shape[2] == 3:
        # Convert to (1, 3, H, W)
        gt_n = gt_n.permute(2, 0, 1).unsqueeze(0)
        pred_n = pred_n.permute(2, 0, 1).unsqueeze(0)
    
    # Calculate dot product
    dot_product = (gt_n * pred_n).sum(1).clamp(-1, 1)
    error_map = torch.acos(dot_product)  # radians
    angular_map = error_map * 180.0 / math.pi  # degrees
    
    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        angular_map = angular_map * mask.squeeze(1)
    
    # Calculate statistics
    if mask is not None:
        valid = mask.sum()
        ang_valid = angular_map[mask.squeeze(1).bool()]
    else:
        valid = angular_map.numel()
        ang_valid = angular_map.flatten()
    
    n_err_mean = ang_valid.sum() / valid
    n_err_med = ang_valid.median()
    
    # Generate color map
    angular_map_colored = colorMap(angular_map.cpu().squeeze().numpy())
    
    n_acc_15 = (ang_valid < 15.0).sum().float() / valid.float()
    n_acc_30 = (ang_valid < 30.0).sum().float() / valid.float()
    
    value = {
        'n_err_mean': n_err_mean.item(),
        'n_err_med': n_err_med.item(),
        'n_acc_15': n_acc_15.item(),
        'n_acc_30': n_acc_30.item(),
    }
    
    return value, angular_map_colored

def generate_error_maps():
    """Generate error maps for all estimated normals"""
    
    # Paths
    est_normal_dir = 'data/DiLiGenT/estNormalNonLambert'
    gt_dir = 'data/DiLiGenT/pmsdata'
    output_dir = 'data/DiLiGenT/errorMaps'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Object names
    objects = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG', 
               'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']
    
    # Method names (extracted from file naming pattern)
    methods = ['ACCV10Wu', 'CVPR08Alldrin', 'CVPR10Higo', 'CVPR12Ikehata', 
               'CVPR12Shi', 'CVPR14Ikehata', 'ECCV12Shi', 'ICCV05Goldman', 'l2']
    
    # Statistics storage
    all_stats = {}
    
    print("Generating error maps for DiLiGenT dataset...")
    print("=" * 60)
    
    for obj in objects:
        print(f"\nProcessing object: {obj}")
        
        # Load ground truth
        gt_path = os.path.join(gt_dir, obj, 'Normal_gt.mat')
        if not os.path.exists(gt_path):
            print(f"  Warning: Ground truth not found at {gt_path}")
            continue
        
        gt_data = loadmat(gt_path)
        gt_normal = gt_data['Normal_gt']
        
        # Load mask
        mask_path = os.path.join(gt_dir, obj, 'mask.png')
        if os.path.exists(mask_path):
            import cv2
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.float32)
        else:
            mask = np.ones((gt_normal.shape[0], gt_normal.shape[1]), dtype=np.float32)
        
        all_stats[obj] = {}
        
        for method in methods:
            # Load estimated normal
            est_filename = f"{obj}_Normal_{method}.mat"
            est_path = os.path.join(est_normal_dir, est_filename)
            
            if not os.path.exists(est_path):
                print(f"  Warning: {est_filename} not found")
                continue
            
            est_data = loadmat(est_path)
            # Try different possible keys in the mat file
            if 'Normal_est' in est_data:
                est_normal = est_data['Normal_est']
            elif f'{obj}_Normal_{method}' in est_data:
                est_normal = est_data[f'{obj}_Normal_{method}']
            else:
                # Get the first non-metadata key
                keys = [k for k in est_data.keys() if not k.startswith('__')]
                if keys:
                    est_normal = est_data[keys[0]]
                else:
                    print(f"  Warning: Could not find normal data in {est_filename}")
                    continue
            
            # Calculate error and generate error map
            stats, error_map = calNormalAcc(gt_normal, est_normal, mask)
            
            # Save error map
            error_map_filename = f"errormap_{obj}_{method}.png"
            error_map_path = os.path.join(output_dir, error_map_filename)
            vutils.save_image(error_map, error_map_path)
            
            # Store statistics
            all_stats[obj][method] = stats
            
            print(f"  {method}: Mean={stats['n_err_mean']:.2f}°, "
                  f"Median={stats['n_err_med']:.2f}°, "
                  f"<15°={stats['n_acc_15']*100:.1f}%, "
                  f"<30°={stats['n_acc_30']*100:.1f}%")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, 'error_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Error Statistics for DiLiGenT Dataset\n")
        f.write("=" * 80 + "\n\n")
        
        for obj in objects:
            if obj not in all_stats:
                continue
            f.write(f"\n{obj}:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Method':<20} {'Mean(°)':<10} {'Median(°)':<10} {'<15°(%)':<10} {'<30°(%)':<10}\n")
            f.write("-" * 80 + "\n")
            
            for method in methods:
                if method in all_stats[obj]:
                    stats = all_stats[obj][method]
                    f.write(f"{method:<20} {stats['n_err_mean']:>8.2f}  "
                           f"{stats['n_err_med']:>8.2f}  "
                           f"{stats['n_acc_15']*100:>8.1f}  "
                           f"{stats['n_acc_30']*100:>8.1f}\n")
    
    print("\n" + "=" * 60)
    print(f"Error maps saved to: {output_dir}")
    print(f"Statistics saved to: {stats_file}")
    print("Done!")

if __name__ == '__main__':
    generate_error_maps()
