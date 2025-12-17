import matplotlib.pyplot as plt
import numpy as np
import sys
import utils as ut

n = 100  # 격자 크기
num_per_class = 20

dot_list = ut.generate_dots(n, num_per_class, num_classes=3)
class_colors = {1: 'red', 2: 'blue', 3: 'green'}

fig, ax = plt.subplots(figsize=(6, 6))

for frame in range(30):
    ax.clear()
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

    for cls, color in class_colors.items():
        idx = np.where(grid == cls)

        ax.scatter(idx[0], idx[1], color=color, s=1, alpha=0.75, label=f'Class {cls}')
    for x, y, c in dot_list:
        ax.scatter(x, y, color=class_colors[c], s=48, edgecolor='black', marker='o', zorder=3)

    ax.set_title(f"K-최근접 이웃 분류 경계 (k = {k})")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.legend(loc='upper right')
    plt.pause(0.2)

plt.show()