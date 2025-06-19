import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from IPython import display
from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# Setup
num_workers = 4
batch_size = 128
home_path = os.getcwd()
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fix seed
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Accuracy
def get_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Training function to record step-wise loss
def train_per_step(model_class, lr=1e-3, epochs_n=5, runs=3):
    step_losses_all_runs = []
    train_loader = get_cifar_loader(train=True)

    for run in range(runs):
        print(f"Run {run+1}/{runs} for {'BN' if model_class==VGG_A_BatchNorm else 'no BN'} at lr={lr}")
        set_random_seeds(2020 + run, device)
        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        step_losses = []
        for epoch in range(epochs_n):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                step_losses.append(loss.item())

        # 保存模型
        model_type = 'bn' if model_class == VGG_A_BatchNorm else 'nobn'
        model_filename = f'model_{model_type}_lr{str(lr).replace(".", "_")}_run{run+1}.pth'
        torch.save(model.state_dict(), os.path.join(models_path, model_filename))
        
        step_losses_all_runs.append(step_losses)

    return step_losses_all_runs

# Try multiple learning rates
learning_rates = [1e-3, 2e-3, 1e-4, 5e-4]

for lr in learning_rates:
    losses_nobn = train_per_step(VGG_A, lr=lr, epochs_n=5, runs=3)
    losses_bn = train_per_step(VGG_A_BatchNorm, lr=lr, epochs_n=5, runs=3)

    # Pad sequences
    def pad_sequences(seqs):
        max_len = max(len(s) for s in seqs)
        return np.array([s + [s[-1]] * (max_len - len(s)) for s in seqs])

    losses_nobn = pad_sequences(losses_nobn)
    losses_bn = pad_sequences(losses_bn)

    # Compute min/max/avg
    min_nobn = np.min(losses_nobn, axis=0)
    max_nobn = np.max(losses_nobn, axis=0)
    avg_nobn = np.mean(losses_nobn, axis=0)

    min_bn = np.min(losses_bn, axis=0)
    max_bn = np.max(losses_bn, axis=0)
    avg_bn = np.mean(losses_bn, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    steps = np.arange(len(avg_nobn))
    plt.fill_between(steps, min_nobn, max_nobn, color='green', alpha=0.3, label='Standard VGG')
    plt.plot(steps, avg_nobn, color='green')
    plt.fill_between(steps, min_bn, max_bn, color='red', alpha=0.3, label='Standard VGG + BatchNorm')
    plt.plot(steps, avg_bn, color='red')
    plt.xlabel('Steps')
    plt.ylabel('Loss Landscape')
    plt.title(f'Loss Landscape at lr={lr}')
    plt.legend()
    plt.grid(True)
    save_name = f'vgg_vs_bn_loss_landscape_lr{str(lr).replace(".", "_")}.png'
    plt.savefig(os.path.join(figures_path, save_name))
    plt.close()
