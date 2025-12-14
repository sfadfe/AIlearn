import numpy as np
import csv


# ==================== 데이터 로드 및 생성 ====================

def train_test_split(x, y, test_size=0.2, random_state=None):
    """
    데이터를 train/test로 분할
    
    Args:
        x, y: 데이터
        test_size: 테스트 데이터 비율
        random_state: 랜덤 시드
    
    Returns:
        x_train, x_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n = len(x)
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]

def generate_random_polynomial(max_degree=5):
    """
    랜덤한 다항 함수 생성
    
    Args:
        max_degree: 최대 차수
    
    Returns:
        degree: 선택된 차수
        coeffs: 계수 리스트 [a0, a1, a2, ..., an]
    """
    # 1차부터 max_degree 사이 랜덤 선택
    degree = np.random.randint(1, max_degree + 1)
    
    # 계수 랜덤 생성 (-5 ~ 5 범위)
    coeffs = np.random.uniform(-5, 5, degree + 1)
    
    return degree, coeffs


def evaluate_polynomial(x, coeffs):
    """다항 함수 계산: y = a0 + a1*x + a2*x^2 + ... + an*x^n"""
    result = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        result += coeff * (x ** i)
    return result


def generate_noisy_data(coeffs, x_range=(0, 10), num_points=100, noise_level=1.0):
    """
    다항 함수 근처에 노이즈가 있는 데이터 생성
    
    Args:
        coeffs: 다항 함수 계수
        x_range: x 범위 (min, max)
        num_points: 생성할 점의 개수
        noise_level: 노이즈 크기 (표준편차)
    
    Returns:
        x_data, y_data: 생성된 데이터
    """
    # x 값 생성 (균등 분포)
    x_data = np.random.uniform(x_range[0], x_range[1], num_points)
    x_data = np.sort(x_data)  # 정렬
    
    # 다항 함수 계산
    y_clean = evaluate_polynomial(x_data, coeffs)
    
    # 노이즈 추가 (정규 분포)
    noise = np.random.normal(0, noise_level, num_points)
    y_data = y_clean + noise
    
    return x_data, y_data


def save_to_csv(x_data, y_data, filename='data.csv'):
    """데이터를 CSV 파일로 저장"""
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])  # 헤더
        for x, y in zip(x_data, y_data):
            writer.writerow([x, y])
    
    print(f"Data saved to {filename}")


def generate_test_data(
    output_file='data.csv',
    max_degree=5,
    x_range=(0, 10),
    num_points=100,
    noise_level=1.0
):
    """
    랜덤 다항 함수를 생성하고 노이즈가 있는 테스트 데이터를 CSV로 저장
    
    Args:
        output_file: 출력 CSV 파일명
        max_degree: 최대 차수
        x_range: x 값 범위
        num_points: 생성할 점의 개수
        noise_level: 노이즈 수준
    """
    # 랜덤 다항 함수 생성
    degree, coeffs = generate_random_polynomial(max_degree)
    
    print(f"Generated polynomial of degree {degree}")
    print("Coefficients:", coeffs)
    
    # 다항 함수 출력
    poly_str = "y = "
    terms = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coeff:.3f}")
        elif i == 1:
            terms.append(f"{coeff:.3f}*x")
        else:
            terms.append(f"{coeff:.3f}*x^{i}")
    poly_str += " + ".join(terms)
    print(f"Polynomial: {poly_str}")
    
    # 노이즈 데이터 생성
    x_data, y_data = generate_noisy_data(coeffs, x_range, num_points, noise_level)
    
    # CSV 저장
    save_to_csv(x_data, y_data, output_file)
    
    print(f"Generated {num_points} data points in range {x_range}")
    print(f"Noise level: {noise_level}")


def load_csv_data(filepath):
    """CSV 파일에서 x, y 좌표 데이터 로드"""
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 헤더 스킵
        for row in reader:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
    
    return np.array(x_data), np.array(y_data)


def polynomial_predict(x, coeffs):
    """다항 함수로 예측: y = a0 + a1*x + a2*x^2 + ... + an*x^n"""
    result = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        result += coeff * (x ** i)
    return result


def mse_loss(y_true, y_pred):
    """평균 제곱 오차 (Mean Squared Error) 계산"""
    return np.mean((y_true - y_pred) ** 2)


def compute_gradients(x, y_true, y_pred, degree):
    """각 계수에 대한 gradient 계산"""
    m = len(x)
    gradients = []
    
    for i in range(degree + 1):
        # ∂L/∂ai = (2/m) * Σ(y_pred - y_true) * x^i
        grad = (2 / m) * np.sum((y_pred - y_true) * (x ** i))
        gradients.append(grad)
    
    return np.array(gradients)


def gradient_descent(x, y, degree, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    """
    경사하강법으로 다항 함수 계수 학습
    
    Returns:
        coeffs_history: 학습 과정의 계수들 [(iteration, coeffs, loss), ...]
    """
    # 계수 초기화 (랜덤)
    coeffs = np.random.randn(degree + 1) * 0.01
    
    coeffs_history = []
    prev_loss = float('inf')
    
    for iteration in range(iterations):
        # 예측
        y_pred = polynomial_predict(x, coeffs)
        
        # 손실 계산
        loss = mse_loss(y, y_pred)
        
        # 기록 (10번마다 또는 마지막)
        if iteration % 10 == 0 or iteration == iterations - 1:
            coeffs_history.append((iteration, coeffs.copy(), loss))
        
        # 조기 종료 (손실 변화가 작으면)
        if abs(prev_loss - loss) < tolerance:
            coeffs_history.append((iteration, coeffs.copy(), loss))
            break
        
        prev_loss = loss
        
        # Gradient 계산 및 업데이트
        gradients = compute_gradients(x, y, y_pred, degree)
        coeffs -= learning_rate * gradients
    
    return coeffs_history


def find_best_degree(x, y, max_degree=10, learning_rate=0.01, iterations=1000, validation_split=0.2):
    """
    최적의 다항식 차수 찾기 (Train/Validation Split 사용)
    
    Args:
        validation_split: 검증 데이터 비율
    
    Returns:
        best_degree: 최적 차수
        all_results: {degree: (final_coeffs, train_loss, val_loss)}
    """
    # 데이터 분할
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)
    
    all_results = {}
    
    for degree in range(1, max_degree + 1):
        # 학습 데이터로 학습
        history = gradient_descent(x_train, y_train, degree, learning_rate, iterations)
        final_iter, final_coeffs, train_loss = history[-1]
        
        # 검증 데이터로 평가
        y_val_pred = polynomial_predict(x_val, final_coeffs)
        val_loss = mse_loss(y_val, y_val_pred)
        
        all_results[degree] = (final_coeffs, train_loss, val_loss)
        
        print(f"Degree {degree}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # 검증 손실이 가장 낮은 차수 선택
    best_degree = min(all_results.keys(), key=lambda d: all_results[d][2])
    
    return best_degree, all_results
