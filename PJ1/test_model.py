import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP([784, 640, 10], 'ReLU', [1e-4, 1e-4])
model.load_model(r'./best_models/best_model.pickle')

test_images_path = r'./dataset/MNIST/t10k-images-idx3-ubyte.gz'
test_labels_path = r'./dataset/MNIST/t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()
print(test_imgs.shape)
#test_imgs = test_imgs.reshape(-1, 1, 28, 28)
print(test_imgs.shape)
logits = model(test_imgs)

print(nn.metric.accuracy(logits, test_labs))