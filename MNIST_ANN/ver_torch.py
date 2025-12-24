import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.font_manager as fm
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if fm.findSystemFonts(fontpaths=['/usr/share/fonts/truetype/nanum/']):
    plt.rc('font', family='NanumGothic')
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    print('NanumGothic 폰트가 시스템에 없습니다. 한글이 깨질 수 있습니다.')

INPUT_SIZE = 784    # 28x28 픽셀 이미지이므로 입력은 784개
HIDDEN_SIZE = 256   # 은닉층 노드 수 (임의 설정)
OUTPUT_SIZE = 10    # 0~9까지 숫자니까 출력은 10개
LEARNING_RATE = 0.01
EPOCHS = 5

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = x.view(-1, 784)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleANN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

print("학습 시작")
for i in range(EPOCHS):
    for j, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{i+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("학습 완료...")

def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))  # (1, 1, 28, 28)
        probabilities = torch.softmax(output, dim=1).cpu().numpy().flatten()
        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3_probs = probabilities[top3_idx]
        return top3_idx, top3_probs, probabilities

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
example_img, example_label = next(iter(test_loader))

top3_idx, top3_probs, all_probs = predict(model, example_img[0])

plt.imshow(example_img[0].squeeze(), cmap='gray')
plt.axis('off')

for i in range(3):
    print(f"{top3_idx[i]} (확률: {top3_probs[i]*100:.2f}%)")

plt.show()