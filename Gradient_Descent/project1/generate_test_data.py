import sys
sys.path.append('.')
import utils as ut

if __name__ == "__main__":
    # 테스트 데이터 생성
    ut.generate_test_data(
        output_file='data.csv',
        max_degree=5,
        x_range=(0, 10),
        num_points=300,
        noise_level=0.5
    )