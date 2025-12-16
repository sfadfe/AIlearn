def gradient_descent(x, y, degree, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    """
    계수(coeffs) 무작위 초기화 (정규분포, 표준편차 0.1)

    반복(iterations):
        ├─ 예측값 계산 (polynomial_predict)
        ├─ 손실(MSE) 계산
        ├─ 10회마다(또는 마지막) 계수/손실 기록
        ├─ 손실 변화가 tolerance 이하이면 조기 종료
        ├─ Gradient 계산 (compute_gradients)
        └─ 계수 업데이트 (경사하강법)
        
    학습 과정 기록(coeffs_history) 반환
    """