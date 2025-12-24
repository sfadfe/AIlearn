import cupy as cp
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from ANN_Model import ANN_class
from tqdm import tqdm
import pickle

input_size = 784
hidden_sizes = [128, 64]
output_size = 10
batch_size = 100
learning_rate = 0.1
epoch = 8

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ANN_class(input_size, hidden_sizes, output_size)

def preprocess_batch(images, labels, num_classes=10):
    x_batch = cp.array(images.view(images.shape[0], -1).numpy())
    t_batch = cp.eye(num_classes)[labels.numpy()]
    return x_batch, t_batch

for i in range(epoch):
    loss = 0
    acc = 0
    count = 0
    
    for img, label in tqdm(train_loader):
        x, t = preprocess_batch(img, label)

        y = model.forward(x)
        
        grads = model.gradient(x, t)
        for key in model.params.keys():
            model.params[key] -= learning_rate * grads[key]
        
        y = model.softmax(y)
        acc += model.accuracy(x, t, y)
        loss += model.loss(x, t, y)
        count += 1
        
    print(f"Epoch {i+1} | Loss: {loss/count:.4f} | Acc: {acc/count:.4f}")

## 모델 저장하기: pickle 라이브러리 사용: cp --> np 배열 (Gpu 메모리 --> CPU 메모리)

params_to_save = {}
for key, val in model.params.items():
    params_to_save[key] = cp.asnumpy(val)

# 2. 파일로 저장 (wb: write binary)
with open('model.pkl', 'wb') as f:
    pickle.dump(params_to_save, f)

print("학습 종료")