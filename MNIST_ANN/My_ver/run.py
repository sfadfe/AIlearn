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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.imshow(img_tensor.squeeze(), cmap='gray')
ax1.axis('off')
ax1.set_title("img")

top3_digits = [k.item() for k in top3]
top3_probs = [probs[0][k].item() * 100 for k in top3]
colors = ['#000000', '#ff0000', '#1c00ff']

bars = ax2.barh(range(3), top3_probs, color=colors, height=0.3)
ax2.set_yticks(range(3))
ax2.set_yticklabels([f"Num {d}" for d in top3_digits], fontsize=12)
ax2.set_xlabel('Probability (%)', fontsize=12)
ax2.set_xlim(0, 100)
ax2.set_title("Top 3 Predictions", fontsize=14)

for bar, prob in zip(bars, top3_probs):
    ax2.text(prob + 1, bar.get_y() + bar.get_height()/2, 
             f'{prob:.2f}%', 
             va='center', fontsize=10, color='black')

ax2.invert_yaxis()
plt.tight_layout()
plt.savefig('result.png', dpi=300)