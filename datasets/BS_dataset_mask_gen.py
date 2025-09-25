import cv2
import numpy as np

# 读取 normal 图 (BGR 格式)
normal = cv2.imread("normal.png")

# 背景灰值
bg_val = np.array([127, 127, 127])

# 允许的偏差 (避免JPEG等压缩带来的噪声)
tol = 110

# 计算绝对差
diff = np.abs(normal.astype(np.int16) - bg_val)

# 判断是否为背景：三个通道都接近128
mask_bg = np.all(diff <= tol, axis=-1)

# 生成mask：前景=255, 背景=0
mask = np.where(mask_bg, 0, 255).astype(np.uint8)

# 保存结果
cv2.imwrite("Mask.png", mask)
