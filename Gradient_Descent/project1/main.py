import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import utils as ut
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--route', type=str, required=True, help='데이터 파일명 (예: data.csv)')
args = parser.parse_args()

route = args.route
def visualize_training(x_norm, y_norm, x_orig, y_orig, degree, learning_rate=0.01, iterations=1000):
    
    print(f"\nTraining polynomial of degree {degree}...")
    coeffs_history = ut.gradient_descent(x_norm, y_norm, degree, learning_rate, iterations)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x_range_orig = np.linspace(x_orig.min(), x_orig.max(), 300)
    # 예측을 위한 정규화
    x_range_norm = (x_range_orig - x_orig.min()) / (x_orig.max() - x_orig.min())
    
    def init():
        ax1.clear()
        ax2.clear()
        ax1.scatter(x_orig, y_orig, c='blue', alpha=0.5, label='Data')  # 원본 좌표
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
        
        ax1.clear()
        ax1.scatter(x_orig, y_orig, c='blue', alpha=0.5, label='Data', s=30)  # 원본 좌표
        
        # 정규화된 x로 예측 후 역정규화
        y_pred_norm = ut.polynomial_predict(x_range_norm, coeffs)
        y_pred_orig = y_pred_norm * (y_orig.max() - y_orig.min()) + y_orig.min()  # y 역정규화
        ax1.plot(x_range_orig, y_pred_orig, 'r-', linewidth=2, label=f'Fitted (iter {iteration})')
        
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
    
    final_iter, final_coeffs, final_loss = coeffs_history[-1]
    print(f"\nFinal Loss: {final_loss:.6f}")
    print(f"Final Coefficients: {final_coeffs}")


csv_file = f"data/{route}" # 데이터 파일 경로.

x, y = ut.load_csv_data(csv_file)
print(f"Loaded {len(x)} data points from {csv_file}")

print("\n=== Normalizing Features ===")
print(f"Original x range: [{x.min():.2f}, {x.max():.2f}]")
print(f"Original y range: [{y.min():.2f}, {y.max():.2f}]")

x_normalized, x_min, x_max = ut.normalize_features(x)
y_normalized, y_min, y_max = ut.normalize_features(y)

print(f"Normalized x range: [{x_normalized.min():.2f}, {x_normalized.max():.2f}]")
print(f"Normalized y range: [{y_normalized.min():.2f}, {y_normalized.max():.2f}]")

# 차수별 학습 및 최적 차수 찾기
print("\n=== Finding Best Polynomial Degree ===")
learning_rate = 0.05  #학습률
iterations = 8000  #사용할 데이터 수

best_degree, all_results = ut.find_best_degree(
    x_normalized, y_normalized,  #계산은 정규화된 데이터를 사용한다.
    learning_rate=learning_rate,
    iterations=iterations,
    validation_split=0.4,
    patience=8
)

print(f"\n=== Best Degree: {best_degree} ===")
print(f"Best Validation Loss: {all_results[best_degree][2]:.6f}")
print(f"Train Loss: {all_results[best_degree][1]:.6f}")
print(f"Full Data Loss: {all_results[best_degree][3]:.6f}")

print("\n=== Training on full dataset ===")
visualize_training(x_normalized, y_normalized, x, y, best_degree, learning_rate, iterations)