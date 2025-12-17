import numpy as np
import sys
from collections import Counter
import matplotlib
matplotlib.rc('font', family='NanumGothic', size=12)
import matplotlib.pyplot as plt

def generate_dots(n, num_per_class, num_classes=3):
    temp = set()
    while len(temp) < num_per_class * num_classes:
        x = np.random.randint(n//5, 4*n//5)
        y = np.random.randint(n//5, 4*n//5)
        temp.add((x, y))
    temp = list(temp)
    dot_list = []
    for c in range(num_classes):
        for i in range(num_per_class):
            idx = c * num_per_class + i
            dot_list.append([temp[idx][0], temp[idx][1], c+1])
    return dot_list

def Minkowski_distance(point1:list , point2:list, p:int): ##민코프스키 거리 계산.
    if len(point1) != 2 or (len(point2) != 3 and len(point2) !=2):
        raise ValueError("=====점의 좌표가 아님..=====")
    
    distance = sum(abs(a - b) ** p for a, b in zip(point1, point2)) ** (1/p)
    return distance

def classification(point1, dot_list, k:int, p:int): #분류
    list_distance = []

    if not all(len(dot) == 3 for dot in dot_list):
        raise ValueError("=====잘못된 점 리스트..=====")
    
    for i in dot_list:
        class_name = i[-1]
        coordinate = [i[0], i[1]]
        distance = Minkowski_distance(point1, coordinate, p)
        list_distance.append((distance, class_name))

    list_distance.sort(key = lambda x: x[0])
    neighbors = [i[1] for i in list_distance[:k]]
    count = Counter(neighbors)
    max_count = max(count.values())
    candidates = [cls for cls, cnt in count.items() if cnt == max_count]

    for i, cls in list_distance[:k]:
        if cls in candidates:
            return cls