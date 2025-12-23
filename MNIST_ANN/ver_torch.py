import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

print("학습 시작...")
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # [Step 1] Forward
        outputs = model(images)
        
        # [Step 2] Loss 계산
        loss = criterion(outputs, labels)
        
        # [Step 3] Backward (기울기 계산)
        optimizer.zero_grad()
        loss.backward()
        
        # [Step 4] Update (가중치 갱신)
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')

print("학습 완료...")