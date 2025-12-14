import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import utils as ut


def visualize_training(x, y, degree, learning_rate=0.01, iterations=1000):
    """경사하강법 학습 과정을 애니메이션으로 시각화"""
    
    # 학습 실행
    print(f"\nTraining polynomial of degree {degree}...")
    coeffs_history = ut.gradient_descent(x, y, degree, learning_rate, iterations)
    
    # 애니메이션 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # x 범위 확장 (부드러운 곡선)
    x_range = np.linspace(x.min(), x.max(), 300)
    
    def init():
        ax1.clear()
        ax2.clear()
        ax1.scatter(x, y, c='blue', alpha=0.5, label='Data')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Polynomial Regression (Degree {degree})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (MSE)')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)
        
        return []
    
    def update(frame_idx):
        if frame_idx >= len(coeffs_history):
            frame_idx = len(coeffs_history) - 1
        
        iteration, coeffs, loss = coeffs_history[frame_idx]
        
        # 왼쪽: 데이터와 근사 곡선
        ax1.clear()
        ax1.scatter(x, y, c='blue', alpha=0.5, label='Data', s=30)
        
        y_pred = ut.polynomial_predict(x_range, coeffs)
        ax1.plot(x_range, y_pred, 'r-', linewidth=2, label=f'Fitted (iter {iteration})')
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(f'Polynomial Regression (Degree {degree})\nLoss: {loss:.6f}', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 오른쪽: 손실 변화
        ax2.clear()
        iterations_so_far = [h[0] for h in coeffs_history[:frame_idx+1]]
        losses_so_far = [h[2] for h in coeffs_history[:frame_idx+1]]
        
        ax2.plot(iterations_so_far, losses_so_far, 'g-', linewidth=2)
        ax2.scatter([iteration], [loss], c='red', s=100, zorder=5)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.set_title('Training Loss Over Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        return []
    
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(coeffs_history),
        init_func=init,
        interval=100,
        repeat=False,
        blit=False
    )
    
    plt.tight_layout()
    plt.show()
    
    # 최종 결과 출력
    final_iter, final_coeffs, final_loss = coeffs_history[-1]
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Final Coefficients: {final_coeffs}")


# CSV 파일 경로
csv_file = "data.csv"

# 데이터 로드
try:
    x, y = ut.load_csv_data(csv_file)
    print(f"Loaded {len(x)} data points from {csv_file}")
except FileNotFoundError:
    print(f"File {csv_file} not found. Generating sample data...")
    # 샘플 데이터 생성 (테스트용)
    x = np.linspace(0, 10, 50)
    y = 2 + 3*x + 0.5*x**2 - 0.1*x**3 + np.random.randn(50) * 2

# 차수별 학습 및 최적 차수 찾기
print("\n=== Finding Best Polynomial Degree ===")
max_degree = 10
learning_rate = 0.0001
iterations = 5000

best_degree, all_results = ut.find_best_degree(
    x, y, 
    max_degree=max_degree,
    learning_rate=learning_rate,
    iterations=iterations
)

print(f"\n=== Best Degree: {best_degree} ===")
print(f"Best Validation Loss: {all_results[best_degree][2]:.6f}")

# 최적 차수로 전체 데이터에 대해 학습 과정 애니메이션
print("\n=== Training on full dataset ===")
visualize_training(x, y, best_degree, learning_rate, iterations)
