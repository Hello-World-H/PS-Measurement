## Structure
- MF-PSN + SFS

## env
- PyTorch  1.11.0
- Python  3.8(ubuntu20.04)
- CUDA  11.3

## TODO
- [x] 测试后的输出图像显示
- [ ] 看一下MF-PSN的论文和代码
- [ ] 为什么面对稀疏输入效果不好？看原始论文，SFS
- [x] 在DiLiGenT 102数据集上测试
  - [x] 修复_getmask
    - [x] 修复精度错误
      - [ ] 如何查看.mat数据？python/matlab
- [ ] 在Bunny&Sphere 数据集测试
- [ ] 消融实验
- [ ] 调参 $\lambda$
- [ ] 加入注意力机制？