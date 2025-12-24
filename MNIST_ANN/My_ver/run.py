import pickle
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from ANN_Model import ANN_class

model = ANN_class(input_size=784, hidden_sizes=[128, 64], output_size=10)

with open('model.pkl', 'rb') as f:
    params = pickle.load(f)

for key, val in params.items():
    model.params[key] = cp.array(val)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

idx = np.random.randint(len(test_dataset))
img_tensor, label = test_dataset[idx]

x = img_tensor.view(1, -1).numpy()
x = cp.array(x)

y = model.forward(x)
probs = model.softmax(y)

sorted_idx = cp.argsort(probs[0])
top3 = sorted_idx[-3:][::-1]


print(f"정답(Label): {label}")

for i, k in enumerate(top3):
    digit = k.item()
    conf = probs[0][k].item() * 100

    print(f"{i+1}위: 숫자 {digit} ({conf:.2f}%)")

plt.imshow(img_tensor.squeeze(), cmap='gray')
plt.axis('off')
plt.show()