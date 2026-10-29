# 车牌识别在雨雾天气下的鲁棒性

车牌识别模型在晴天数据上表现很好，一遇雨雾就崩。这个项目做的是：**先用代码合成恶劣天气，再把模型在这些数据上重训一遍**。

## 思路

1. 用 `cv2` 对原始车牌图片做两种退化：
   - **雾**：`addWeighted` 把图像与白色图层按强度混合，模拟能见度下降
   - **雨**：随机生成雨滴线并叠加，模拟雨幕遮挡
2. 把合成数据并入 CCPD2020 绿牌训练集（**5769 张**），保留原始验证集作为对照。
3. 用 LPRNet 重新训练，保留 2000 轮、4000 轮与最终三个检查点。

## 本仓库内容

```
lprnet-rain-fog.ipynb   天气合成的实现 + 数据路径配置 + 训练启动
test_LPRNet.py          推理脚本（载入权重、读出车牌字符）
weights/
  LPRNet_iter_2000.pth
  LPRNet_iter_4000.pth
  Final_LPRNet_model.pth
```

## 诚实说明

- 网络结构与训练主循环来自开源项目 **LPRNet_Pytorch**（作者 aiboy.wei）。我做的是**天气合成 + 数据组织 + 训练与检查点管理**，不是从零写 LPRNet。
- 训练是通过命令行跑的，所以 notebook 里没有完整的逐 epoch 日志；能验证的结果是三个检查点本身与数据集规模。
- 具体的识别准确率需要把权重加载回来、在 CCPD2020 测试集上跑一遍才能给，我没有留存那组数字，所以不在这里写。

## 复现

```bash
pip install torch torchvision opencv-python numpy
# 1. 准备 CCPD2020 绿牌数据（train / val）
# 2. 跑 notebook 里的天气合成，生成增强后的训练目录
# 3. 用上游 LPRNet_Pytorch 的 train_LPRNet.py 指向该目录，加载 weights/ 里的检查点继续训练或直接推理
python test_LPRNet.py --test_img_dirs <测试集> --pretrained_model weights/Final_LPRNet_model.pth
```

`results/` 里是 notebook 输出的一张图。
