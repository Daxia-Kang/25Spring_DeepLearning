import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm 
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
num_workers = 4
batch_size = 128

# add our package dir to path 
home_path = os.getcwd()
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# Ensure output directories exist
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

# Make sure you are using the right device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(torch.cuda.current_device()))


# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    ## --------------------
    # Add code as needed
    #
    #
    #
    #
    ## --------------------
    break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            loss_list.append(loss.item())  
            loss.backward()
            if model.classifier[-1].weight.grad is not None:
                grad.append(model.classifier[-1].weight.grad.norm().item())
            else:
                grad.append(0)
            optimizer.step()
            learning_curve[epoch] += loss.item()

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        learning_curve[epoch] /= batches_n
        axes[0].plot(learning_curve)
        plt.close(f)
        # Test your model and save figure here (not required)
        # Add code as needed
        val_accuracy = get_accuracy(model, val_loader, device)
        val_accuracy_curve[epoch] = val_accuracy
        print(f"Epoch {epoch+1}: Val Accuracy: {val_accuracy:.4f}")

    return losses_list, grads, learning_curve, val_accuracy_curve


# Train your model
# feel free to modify
epo = 20
loss_save_path = figures_path
grad_save_path = figures_path

set_random_seeds(seed_value=2020, device=device)
model = VGG_A()
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
loss, grads, learning_curve, val_accuracy_curve = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
np.savetxt(os.path.join(loss_save_path, 'loss.txt'), loss, fmt='%s', delimiter=' ')
np.savetxt(os.path.join(grad_save_path, 'grads.txt'), grads, fmt='%s', delimiter=' ')

# Maintain two lists: max_curve and min_curve,
# select the maximum value of loss in all models
# on the same step, add it to max_curve, and
# the minimum value to min_curve
min_curve = []
max_curve = []
## --------------------
# Add your code
for epoch_losses in loss:
    if epoch_losses:
        min_curve.append(min(epoch_losses))
        max_curve.append(max(epoch_losses))
## --------------------

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_and_accuracy(learning_curve, val_acc_curve):
    plt.figure(figsize=(10, 6))
    epochs = list(range(len(learning_curve)))

    # 主图（左 y 轴）：Loss
    ax1 = plt.gca()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(epochs, learning_curve, label='Avg Loss', color='blue')
    ax1.fill_between(epochs, min_curve, max_curve, color='gray', alpha=0.3, label='Loss Range')
    ax1.grid(True)

    # 副图（右 y 轴）：Accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy')
    ax2.plot(epochs, val_acc_curve, 'r--', label='Val Accuracy')

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('Loss Range, Average Loss and Validation Accuracy')
    save_path = os.path.join(figures_path, 'loss_landscape.png')
    plt.savefig(save_path)
    print(f"Saved loss landscape to: {save_path}")
    
    plt.close()

plot_loss_and_accuracy(learning_curve, val_accuracy_curve)