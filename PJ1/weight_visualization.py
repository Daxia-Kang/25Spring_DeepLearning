import mynn as nn
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model = nn.models.Model_CNN([1, 4, 16, 2304, 128, 10], 'ReLU', [1e-4] * 5)
model.load_model(r'./best_models/best_model_CNN.pickle')

# ===== 可视化第一层卷积层权重 =====
conv1_weights = model.layers[0].params['W']  # shape: (4, 1, k, k)
num_filters_1 = conv1_weights.shape[0]

plt.figure(figsize=(num_filters_1 * 2, 2))
for i in range(num_filters_1):
    plt.subplot(1, num_filters_1, i + 1)
    plt.imshow(conv1_weights[i, 0], cmap='gray')  # 可视化每个 filter 的通道 0
    plt.title(f"F{i}")
    plt.axis('off')
plt.suptitle("Conv1 Filters")
plt.tight_layout()
plt.savefig("conv1_weights.png")  # 保存第一层权重图像
plt.show()

# ===== 可视化第二层卷积层权重 =====
conv2_weights = model.layers[2].params['W']  # shape: (16, 4, k, k)
num_filters_2 = conv2_weights.shape[0]
in_channels_2 = conv2_weights.shape[1]

# 我们将每个 filter 的每个输入通道画出来，总共 16×4 = 64 张图
plt.figure(figsize=(in_channels_2 * 2, num_filters_2 * 2))
for i in range(num_filters_2):
    for j in range(in_channels_2):
        idx = i * in_channels_2 + j
        plt.subplot(num_filters_2, in_channels_2, idx + 1)
        plt.imshow(conv2_weights[i, j], cmap='gray')
        plt.title(f"F{i}C{j}", fontsize=8)
        plt.axis('off')
plt.suptitle("Conv2 Filters (per input channel)")
plt.tight_layout()
plt.savefig("conv2_weights.png")  # 保存第二层权重图像
plt.show()
