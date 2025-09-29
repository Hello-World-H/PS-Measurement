## Structure
- MF-PSN + SFS

## env
- PyTorch  1.11.0
- Python  3.8(ubuntu20.04)
- CUDA  11.3

## TODO
- [x] 测试后的输出图像显示
- [x] 测试后的误差图显示
- [ ] 看一下MF-PSN的论文和代码
  - [ ] 复现MF-PSN的工作 是否达到指标
  - [ ] MF-PSN是否有人作出改进？
- [ ] 为什么面对稀疏输入效果不好？看原始论文，SFS
- [x] 在DiLiGenT 102数据集上测试
  - [x] 修复_getmask
    - [x] 修复精度错误
      - [x] 如何查看.mat数据？python/matlab
- [x] 在Bunny&Sphere 数据集测试
  - [ ] 其他论文对BS数据集的测试方法
- [ ] 消融实验
- [ ] 调参 $\lambda$
- [ ] 加入注意力机制？

## Testing
`python eval/run_model.py --retrain data/Training/calib/train/checkp_best.pth.tar --in_img_num 96`

## Problem Analysis
### De-performance of the model on sparse input (10@DiLiGenT, from paper)
1. Bear 4.89 -> 5.489
2. Cow 8.41 -> 14.317
3. Avg 8.48 -> 8.677
### Why the performance is not good with sparse input?
1. 丢失半阴影区法线信息
2. 