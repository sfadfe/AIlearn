import sys
sys.path.append('.')
import utils as ut


output_file = "data/data.csv"
max_degree = 5
x_range = (-20, 20)
num_points = 100
noise_level = 11

degree, coeffs = ut.generate_random_polynomial(max_degree)
    
print(f"Generated polynomial of degree {degree}")
print("Coefficients:", coeffs)

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

x_data, y_data = ut.generate_noisy_data(coeffs, x_range, num_points, noise_level)

ut.save_to_csv(x_data, y_data, output_file)
    
print(f"Generated {num_points} data points in range {x_range}")
print(f"Noise level: {noise_level}")