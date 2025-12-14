def Minkowski_distance(point1:list , point2:list, p:int): #민코프스키 거리 계산.
    if len(point1) != 2 or (len(point2) != 3 and len(point2) !=2):
        raise ValueError("=====점의 좌표가 아님..=====")
    
    distance = sum(abs(a - b) ** p for a, b in zip(point1, point2)) ** (1/p)
    return distance

def classification(point1, dot_list, k:int, p:int): #분류
    list_distance = []
    neighbors = []
    if not all(len(dot) == 3 for dot in dot_list):
        raise ValueError("=====잘못된 점 리스트..=====")
    
    for i in dot_list:
        class_name = i[-1]
        coordinate = [i[0], i[1]] # coordinate: 좌표 / 점의 좌표
        distance = Minkowski_distance(point1, coordinate, p)
        list_distance.append((distance, class_name))

    list_distance.sort(key=lambda x: x[0]) #거리 기준 오름차순
    for i in list_distance:
        if i[0] <= list_distance[k-1][0]:
            neighbors.append(i[1])
    
    num_1 = neighbors.count(1)
    num_2 = neighbors.count(2)
    num_3 = neighbors.count(3)

    if num_1 != num_2 and num_2 != num_3 and num_1 != num_3:
        return max((num_1,1), (num_2,2), (num_3,3))[1] #개수 다 다르면 가장 많은 클래스 반환
    elif num_1 == num_2 and num_1 == num_3:
        return neighbors[0] #모두 동수일 경우 가장 가까운 이웃의 클래스 반환
    else:
        num_list = [[num_1, 1], [num_2, 2], [num_3, 3]]
        a = sorted(num_list, key=lambda x: x[0], reverse=True)
        return neighbors[min(neighbors.index(a[0][1]), neighbors.index(a[1][1]))] #동수인 클래스 중 가장 가까운 이웃의 클래스 반환