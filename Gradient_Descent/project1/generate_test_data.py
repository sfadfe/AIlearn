import numpy as np
import csv
import os

def generate_random_polynomial(max_degree):
    degree = np.random.randint(1, max_degree + 1)
    coeffs = np.random.uniform(-5, 5, degree + 1)
    return degree, coeffs

def generate_noisy_data(coeffs, x_range, num_points, noise_level):
    x_data = np.random.uniform(x_range[0], x_range[1], num_points)
    x_data.sort()
    
    y_clean = np.zeros_like(x_data)
    for i, coeff in enumerate(coeffs):
        y_clean += coeff * (x_data ** i)
    
    noise = np.random.normal(0, noise_level, num_points)
    y_data = y_clean + noise
    
    return x_data, y_data

def save_to_csv(x_data, y_data, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y'])
        for x, y in zip(x_data, y_data):
            writer.writerow([x, y])
    print(f"Data saved to {filename}")

def main():
    output_file = "data/data.csv"
    max_degree = 5
    x_range = (-20, 20)
    num_points = 100
    noise_level = 11

    degree, coeffs = generate_random_polynomial(max_degree)
    
    print(f"Generated polynomial of degree {degree}")
    print("Coefficients:", coeffs)

    terms = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coeff:.3f}")
        elif i == 1:
            terms.append(f"{coeff:.3f}x")
        else:
            terms.append(f"{coeff:.3f}x^{i}")
    
    poly_str = " + ".join(terms).replace("+ -", "- ")
    print(f"Polynomial: {poly_str}")

    x_data, y_data = generate_noisy_data(coeffs, x_range, num_points, noise_level)
    save_to_csv(x_data, y_data, output_file)
    
    print(f"Generated {num_points} data points in range {x_range}")
    print(f"Noise level: {noise_level}")

if __name__ == "__main__":
    main()