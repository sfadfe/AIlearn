import matplotlib.pyplot as plt
import numpy as np
import sys
import utils as ut
import argparse

import matplotlib
matplotlib.rc('font', family='NanumGothic')

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=20, help='클래스별 점 개수')
parser.add_argument('--n', type=int, default=100, help='격자 크기')
args = parser.parse_args()

n = args.n
num_per_class = args.num

dot_list = ut.generate_dots(n, num_per_class, num_classes=3)
class_colors = {1: 'red', 2: 'blue', 3: 'green'}


fig, (ax, ax_acc) = plt.subplots(1, 2, figsize=(12, 6))


accuracies = []  # 원 안 정확도
loo_accuracies = []  # leave-one-out 정확도

for frame in range(30):
    ax.clear()
    ax_acc.clear()
    k = frame + 1

    grid = np.zeros((n, n), dtype=int)
    total = n * n

    for i in range(n):
        percent = (i * n) / total
        arrow = '█' * int(round(percent * 30))
        spaces = ' ' * (30 - len(arrow))

        sys.stdout.write(f"\rK = {k}  진행도: [{arrow}{spaces}] {percent*100:5.1f}%")
        sys.stdout.flush()

        for j in range(n):
            grid[i, j] = ut.classification([i, j], dot_list, k, p=2)

    sys.stdout.write(f"\rK = {k}  진행도: [{'█'*30}] 100.0%")
    sys.stdout.flush()

    sys.stdout.write('\r' + ' ' * 50 + '\r')
    sys.stdout.flush()


    acc = ut.accuracy_in_circle(dot_list, grid, r=5)
    accuracies.append(acc)

    loo_acc = ut.leave_one_out_accuracy(dot_list, k, p=2)
    loo_accuracies.append(loo_acc)


    for cls, color in class_colors.items():
        idx = np.where(grid == cls)
        ax.scatter(idx[0], idx[1], color=color, s=1, alpha=0.75, label=f'Class {cls}')
    for x, y, c in dot_list:
        ax.scatter(x, y, color=class_colors[c], s=48, edgecolor='black', marker='o', zorder=3)

    ax.set_title(f"K-최근접 이웃 분류 경계 (k = {k})")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.legend(loc='upper right')

    x_vals = np.arange(1, frame+2)
    ax_acc.plot(x_vals, accuracies[:frame+1], label='원 안 정확도', color='blue', marker='o')
    ax_acc.plot(x_vals, loo_accuracies[:frame+1], label='Leave-One-Out', color='red', marker='o')
    ax_acc.set_title('K별 정확도')
    ax_acc.set_xlabel('K')
    ax_acc.set_ylabel('정확도')
    ax_acc.set_xlim(1, 30)
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()

    plt.tight_layout()
    plt.pause(0.2)

plt.show()