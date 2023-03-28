import numpy as np
import matplotlib.pyplot as plt
import os

fig_dir = 'fig'
log_dir = 'log'

os.makedirs(fig_dir, exist_ok=True)
file_name = "criteo_raw"
train_loss = []
train_auc = []
val_loss = []
val_auc = []
val_score = []

with open(f"{log_dir}/{file_name}.log", "r") as f:
    for line in f.readlines():
        if "binary_crossentropy" not in line:
            continue
        line = line.strip().split(" ")
        # print(line)
        train_loss.append(float(line[8]))
        train_auc.append(float(line[12]))
        val_loss.append(float(line[16]))
        val_auc.append(float(line[20]))
        val_score.append(float(line[23])/2)
        # break

fig, ax = plt.subplots(2, 1,figsize=(8, 12))
ax[0].plot(train_auc, label='train_auc')
ax[0].plot(train_loss, label='train_loss')

ax[0].set_xticks(np.arange(0, 10))
ax[0].set_xticklabels(np.arange(1, 11))
ax[0].legend()
ax[0].grid()
ax[0].set_xlabel("Epoch")

ax[1].plot(val_auc, label='val_auc')
ax[1].plot(val_loss, label='val_loss')

ax[1].plot(val_score, label='val_score', color='red')
ax[1].set_xticks(np.arange(0, 10))
ax[1].set_xticklabels(np.arange(1, 11))
ax[1].legend()
ax[1].grid()
ax[1].set_xlabel("Epoch")

plt.savefig(f"{fig_dir}/{file_name}.png")
