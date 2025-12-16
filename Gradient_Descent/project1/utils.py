import numpy as np
import csv



def normalize_features(x):
    """
    [0, 1]로 특성 정규화함.
        x_normalized: 정규화된 특성
        x_min, x_max: 정규화 파라미터
    """
    x_min, x_max = x.min(), x.max()
    x_normalized = (x - x_min) / (x_max - x_min)
    return x_normalized, x_min, x_max


def denormalize_coeffs(coeffs_normalized, x_min, x_max):
    """
    정규화된 공간에서 학습한 계수를 원본 공간으로 변환
    
    y = a0 + a1*x_norm + a2*x_norm^2 + ...
    x_norm = (x - x_min) / (x_max - x_min)
    
    이를 원본 x에 대한 식으로 변환
    """
    # 간단히 하기 위해 정규화된 계수 그대로 반환
    # 실제로는 복잡한 변환이 필요하지만, 시각화용으로는 정규화된 공간에서 사용
    return coeffs_normalized


def train_test_split(x, y, test_size=0.2, random_state=None):
    """
    데이터를 train/test로 분할
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

def generate_random_polynomial(max_degree):
    """
    랜덤한 다항 함수 생성
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

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])  # 헤더
        for x, y in zip(x_data, y_data):
            writer.writerow([x, y])
    
    print(f"Data saved to {filename}")

def load_csv_data(filepath):
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
    
    return np.array(x_data), np.array(y_data)


def polynomial_predict(x, coeffs):
    #다항 함수로 예측: y = a0 + a1*x + a2*x^2 + ... + an*x^n
    result = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        result += coeff * (x ** i)
    return result


def mse_loss(y_true, y_pred):
    #평균제곱오차를 return함.
    return np.mean((y_true - y_pred) ** 2)


def calculate_aic(mse, n, k):
    """
    AIC (Akaike Information Criterion) 계산

        mse: 평균제곱오차
        n: 데이터 포인트 개수
        k: 파라미터 개수 (차수 + 1)
    
    """
    return n * np.log(mse) + 2 * k


def calculate_bic(mse, n, k):
    """
    BIC (Bayesian Information Criterion) 계산

    식: BIC = n * log(mse) + k * log(n)

        mse: 평균제곱오차
        n: 데이터 포인트 개수
        k: 파라미터 개수 (차수 + 1)
    """

    return n * np.log(mse) + k * np.log(n)


def compute_gradients(x, y_true, y_pred, degree):
    m = len(x)
    gradients = []
    
    for i in range(degree + 1):
        # ∂L/∂ai = (2/m) * Σ(y_pred - y_true) * x^i
        grad = (2 / m) * np.sum((y_pred - y_true) * (x ** i))
        gradients.append(grad)
    
    return np.array(gradients)


def gradient_descent(x, y, degree, learning_rate=0.01, iterations=1000, tolerance=1e-6):

    coeffs = np.random.randn(degree + 1) * 0.1
    
    coeffs_history = []
    prev_loss = float('inf')
    
    for iteration in range(iterations):
        # 예측
        y_pred = polynomial_predict(x, coeffs)
        
        # 손실 계산
        loss = mse_loss(y, y_pred)
        
        if iteration % 10 == 0 or iteration == iterations - 1:
            coeffs_history.append((iteration, coeffs.copy(), loss))
        
        # 조기 종료
        if abs(prev_loss - loss) < tolerance:
            coeffs_history.append((iteration, coeffs.copy(), loss))
            break
        
        prev_loss = loss
        
        # Gradient 계산 및 업데이트
        gradients = compute_gradients(x, y, y_pred, degree)
        coeffs -= learning_rate * gradients
    
    return coeffs_history


def find_best_degree(x, y, learning_rate=0.01, iterations=1000, validation_split=0.3, patience=3):

    #Best validation loss 대비 악화가 patience번 지속되면 조기 중단

    # 데이터 분할
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_split, random_state=42)
    
    all_results = {}
    best_val_loss = float('inf')
    no_improvement_count = 0
    degree = 1
    
    print(f"Starting degree search (patience={patience}, val_split={validation_split})...")
    
    while True:
        # 전체 데이터로 학습
        history = gradient_descent(x, y, degree, learning_rate, iterations)
        final_iter, final_coeffs, full_train_loss = history[-1]
        
        # Train으로 평가
        y_train_pred = polynomial_predict(x_train, final_coeffs)
        train_loss = mse_loss(y_train, y_train_pred)
        
        # Validation으로 평가 (과적합 체크)
        y_val_pred = polynomial_predict(x_val, final_coeffs)
        val_loss = mse_loss(y_val, y_val_pred)
        
        # BIC 계산
        n = len(y_val)
        k = degree + 1  # 파라미터 개수: 고차 다항식의 경우 패널티를 부여한다.
        bic = calculate_bic(val_loss, n, k)
        
        all_results[degree] = (final_coeffs, train_loss, val_loss, full_train_loss, bic)
        
        print(f"Degree {degree}: Train = {train_loss:.6f}, Val = {val_loss:.6f}, BIC = {bic:.2f}")
        
        # 조기 중단 체크 (Best BIC 대비)
        if bic < best_val_loss:
            improvement = best_val_loss - bic
            best_val_loss = bic
            no_improvement_count = 0
            print(f"  → New best BIC! Improved by {improvement:.2f} ✓")
        else:
            no_improvement_count += 1
            gap = bic - best_val_loss
            print(f"  → No improvement (gap: {gap:.2f}, count: {no_improvement_count}/{patience})")
            
            if no_improvement_count >= patience:
                print(f"\n Early stopping at degree {degree}!")
                print(f"No improvement for {patience} consecutive degrees.")
                break
        
        degree += 1
        
        #무한 루프 방지
        if degree > 50:
            print(f"\n Reached maximum degree limit")
            break
    
    # BIC가 가장 낮은 차수 선택
    best_degree = min(all_results.keys(), key=lambda d: all_results[d][4])  # index 4 = BIC
    best_bic = all_results[best_degree][4]
    best_val_loss = all_results[best_degree][2]
    
    print(f"\nTested {len(all_results)} degrees (1 to {max(all_results.keys())})")
    print(f"Best degree: {best_degree} with BIC: {best_bic:.2f}, Val Loss: {best_val_loss:.6f}")
    
    return best_degree, all_results
