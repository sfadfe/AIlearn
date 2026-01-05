import numpy as np
import cupy as cp
import csv

class AdamW:

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = cp.zeros_like(params)
            self.v = cp.zeros_like(params)
        
        self.t += 1
        
        # Weight Decay (L2 Regularization equivalent)
        params = params - self.lr * self.weight_decay * params
        
        # Update moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        params = params - self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)
        
        return params

def _create_poly_features_gpu(x_gpu, degree):
    
    if x_gpu.ndim == 1:
        x_gpu = x_gpu[:, None]
    
    powers = cp.arange(degree + 1)
    return x_gpu ** powers

def _mse_loss_gpu(y_true, y_pred):
    return cp.mean((y_true - y_pred) ** 2)

def load_csv_data(filepath):
    """
    CSV 파일을 읽어 NumPy 배열로 반환 (main.py 호환성 유지)
    """
    x_data = []
    y_data = []
    
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader) # 헤더 건너뛰기
        for row in reader:
            x_data.append(float(row[0]))
            y_data.append(float(row[1]))
            
    return np.array(x_data), np.array(y_data)

def normalize_features(x):
    """
    NumPy 배열을 입력받아 정규화 후 NumPy 배열 반환
    """
    x_min, x_max = x.min(), x.max()
    denom = x_max - x_min
    if denom == 0: denom = 1.0
    x_normalized = (x - x_min) / denom
    return x_normalized, x_min, x_max

def polynomial_predict(x, coeffs):
    """
    예측 함수
    Input: x (NumPy or CuPy), coeffs (NumPy or CuPy)
    Output: NumPy array (for Matplotlib visualization)
    """
    # 입력을 GPU로 이동 (이미 GPU라면 유지)
    x_gpu = cp.asarray(x)
    coeffs_gpu = cp.asarray(coeffs)
    
    degree = len(coeffs) - 1
    X_poly = _create_poly_features_gpu(x_gpu, degree)
    
    # 행렬 곱으로 예측
    y_pred_gpu = cp.dot(X_poly, coeffs_gpu)
    
    # 결과를 CPU(NumPy)로 반환
    return cp.asnumpy(y_pred_gpu)

def gradient_descent(x, y, degree, learning_rate=0.01, iterations=1000):
    """
    AdamW를 이용한 경사 하강법
    Input: x, y (NumPy arrays)
    Output: coeffs_history (list of tuples with CPU data)
    """
    # 1. 데이터를 GPU로 이동
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    m = len(x_gpu)
    
    # 2. Feature 행렬 생성 (한 번만 계산)
    X_poly = _create_poly_features_gpu(x_gpu, degree) # Shape: (N, degree+1)
    
    # 3. 계수 초기화
    # Xavier/He 초기화 스타일 (작은 랜덤 값)
    coeffs_gpu = cp.random.randn(degree + 1) * 0.1
    
    # 4. AdamW Optimizer 초기화
    optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    
    coeffs_history = []
    
    for i in range(iterations):
        # 예측 (행렬 연산)
        y_pred = cp.dot(X_poly, coeffs_gpu)
        
        # Loss 계산
        loss = _mse_loss_gpu(y_gpu, y_pred)
        
        # 기록 (시각화를 위해 일정 간격 혹은 마지막)
        # Matplotlib 처리를 위해 CPU로 데이터 복사 (.get() or cp.asnumpy())
        if i % 10 == 0 or i == iterations - 1:
            coeffs_history.append((
                i, 
                cp.asnumpy(coeffs_gpu), 
                float(loss)
            ))
        
        # Gradient 계산 (Vectorized)
        # dL/dw = (2/m) * X.T @ (y_pred - y)
        gradients = (2 / m) * cp.dot(X_poly.T, (y_pred - y_gpu))
        
        # AdamW 업데이트
        coeffs_gpu = optimizer.step(coeffs_gpu, gradients)
        
    return coeffs_history

def calculate_bic(mse, n, k):
    """BIC 계산 (GPU 사용 가능)"""
    # mse가 0에 가까우면 log에서 -inf가 나오므로 클리핑
    mse = max(mse, 1e-10)
    return n * np.log(mse) + k * np.log(n)

def find_best_degree(x, y, learning_rate=0.01, iterations=1000, validation_split=0.2, patience=3):
    """
    최적 차수 찾기 (GPU 가속 + AdamW)
    """
    # 1. 데이터를 GPU로 이동 및 분할
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    n_samples = len(x)
    
    # 랜덤 셔플
    indices = cp.arange(n_samples)
    cp.random.shuffle(indices)
    split_idx = int(n_samples * (1 - validation_split))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    x_train = x_gpu[train_idx]
    y_train = y_gpu[train_idx]
    x_val = x_gpu[val_idx]
    y_val = y_gpu[val_idx]
    
    all_results = {}
    best_bic = float('inf')
    no_improvement_count = 0
    degree = 1
    
    print(f"Starting GPU search (AdamW, patience={patience})...")
    
    while True:
        # --- 내부 학습 루프 (gradient_descent 함수를 직접 호출하지 않고 최적화하여 구현) ---
        X_train_poly = _create_poly_features_gpu(x_train, degree)
        X_val_poly = _create_poly_features_gpu(x_val, degree)
        
        # 초기화
        coeffs = cp.random.randn(degree + 1) * 0.1
        optimizer = AdamW(learning_rate=learning_rate)
        m_train = len(x_train)
        
        # 학습 (고속 루프)
        for _ in range(iterations):
            y_pred = cp.dot(X_train_poly, coeffs)
            grads = (2 / m_train) * cp.dot(X_train_poly.T, (y_pred - y_train))
            coeffs = optimizer.step(coeffs, grads)
            
        # 평가
        y_train_pred = cp.dot(X_train_poly, coeffs)
        train_loss = float(_mse_loss_gpu(y_train, y_train_pred))
        
        y_val_pred = cp.dot(X_val_poly, coeffs)
        val_loss = float(_mse_loss_gpu(y_val, y_val_pred))
        
        # 전체 데이터에 대한 Loss (보고용)
        # 주의: 이 부분은 근사치를 위해 train loss를 쓸 수도 있지만 정확성을 위해 계산
        X_full_poly = _create_poly_features_gpu(x_gpu, degree)
        y_full_pred = cp.dot(X_full_poly, coeffs)
        full_loss = float(_mse_loss_gpu(y_gpu, y_full_pred))

        # BIC 계산
        n_val = len(y_val)
        k = degree + 1
        bic = calculate_bic(val_loss, n_val, k)
        
        # 결과 저장 (CPU 호환 타입으로 변환)
        all_results[degree] = (
            cp.asnumpy(coeffs), 
            train_loss, 
            val_loss, 
            full_loss, 
            bic
        )
        
        print(f"Degree {degree}: Train={train_loss:.5f}, Val={val_loss:.5f}, BIC={bic:.2f}")
        
        # 조기 종료 체크
        if bic < best_bic:
            best_bic = bic
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(f"Early stopping at degree {degree}")
                break
        
        degree += 1
        if degree > 30: # 안전장치
            break
            
    best_degree = min(all_results.keys(), key=lambda d: all_results[d][4]) # BIC 기준
    
    return best_degree, all_results