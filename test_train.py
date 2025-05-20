# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

import torch
from scipy.ndimage import rotate
def augment_by_rotation(img, shape=(28, 28)):
    img = img.reshape(shape)
    angle = np.random.uniform(-15, 15)  # 随机旋转角度
    rotated_img = rotate(img, angle, reshape=False, mode='nearest')
    return rotated_img.flatten()

# 指定设备为 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx] # 标签数据，存储的是类别标签。
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

augmented_imgs = []
augmented_labs = []

for img, lab in zip(train_imgs, train_labs):
    augmented_imgs.append(img)  # 原图
    augmented_labs.append(lab)

    rotated_img = augment_by_rotation(img)
    augmented_imgs.append(rotated_img)  # 增强图
    augmented_labs.append(lab)

train_imgs = np.array(augmented_imgs)
train_labs = np.array(augmented_labs)

print(f"训练集增强完成，共 {train_imgs.shape[0]} 张图像。")


# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

'''
# reshape for CNN input
train_imgs = train_imgs.reshape(-1, 1, 28, 28)
valid_imgs = valid_imgs.reshape(-1, 1, 28, 28)

CNN_model = nn.models.Model_CNN([1, 4, 16, 2304, 128, 10], 'ReLU', [1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
                        
optimizer = nn.optimizer.SGD(init_lr=0.02, model=CNN_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=CNN_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(CNN_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')
'''
linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 640, 10], 'ReLU')
optimizer = nn.optimizer.MomentumGD(init_lr=0.04, model=linear_model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.savefig("result.png")