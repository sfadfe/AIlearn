import cupy as cp
import numpy as np

class PINNs: 
    def __init__(self, input_size, hidden_sizes, output_size, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
        self.m1, self.m2 = m1, m2
        self.L1, self.L2 = L1, L2
        self.g = g
        
        self.params = {}
        self.hidden_sizes = hidden_sizes
        
        all_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(all_sizes) - 1
        
        for i in range(self.num_layers):
            in_node = all_sizes[i]
            out_node = all_sizes[i+1]

            w_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)
            
            # Tanh activation function에 적합한 Xavier Initialization 적용
            # 표준편차 = sqrt(1 / 입력노드수) 또는 sqrt(2 / (입력+출력))
            scale = cp.sqrt(1.0 / in_node)
            self.params[w_key] = cp.random.randn(in_node, out_node) * scale
            self.params[b_key] = cp.zeros(out_node)
            
        self.grads = {}
        self.cache = {}

    def forward(self, x):
        self.cache = {}
        out = x
        self.cache['z0'] = x 

        for i in range(1, self.num_layers):
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            
        # 마지막 층 (Identity function)
            z = cp.dot(out, W) + b
            self.cache['a' + str(i)] = z # 활성화 전
            
            out = cp.tanh(z)
            self.cache['z' + str(i)] = out # 활성화 후

        # 마지막 층 (Identity function)
        last_idx = self.num_layers
        W_last = self.params['W' + str(last_idx)]
        b_last = self.params['b' + str(last_idx)]
        
        out = cp.dot(out, W_last) + b_last
        
        return out

    def get_energy(self, state):
        th1 = state[:, 0]
        w1  = state[:, 1]
        th2 = state[:, 2]
        w2  = state[:, 3]
        
        # 1. 위치 에너지
        y1 = -self.L1 * cp.cos(th1)
        y2 = y1 - self.L2 * cp.cos(th2)
        V = self.m1 * self.g * y1 + self.m2 * self.g * y2
        
        # 2. 운동 에너지
        v1_sq = (self.L1 * w1)**2
        v2_sq = (self.L1 * w1)**2 + (self.L2 * w2)**2 + \
                2 * self.L1 * self.L2 * w1 * w2 * cp.cos(th1 - th2)
                
        T = 0.5 * self.m1 * v1_sq + 0.5 * self.m2 * v2_sq
        
        return T + V

    def loss(self, x_input, y_pred, t_true): # 손실함수: RK4 데이터에서의 손실 + 물리항에서의 손실(E_실제 - E_예측)

        batch_size = y_pred.shape[0]
        
        loss_data = 0.5 * cp.sum((y_pred - t_true) ** 2) / batch_size
        
        E_in = self.get_energy(x_input)
        E_pred = self.get_energy(y_pred)
        
        loss_physics = 0.5 * cp.sum((E_pred - E_in) ** 2) / batch_size
        
        lambda_p = 0.2
        
        return loss_data + lambda_p * loss_physics

    def backward(self, x, t_true, y_pred):
        batch_size = x.shape[0]
        dout = (y_pred - t_true) / batch_size

        for i in range(self.num_layers, 0, -1):
            W_key = 'W' + str(i)
            b_key = 'b' + str(i)
            
            # 이전 층 입력값 가져오기
            if i == 1:
                z_prev = self.cache['z0']
            else:
                z_prev = self.cache['z' + str(i-1)]

            self.grads[W_key] = cp.dot(z_prev.T, dout)
            
            self.grads[b_key] = cp.sum(dout, axis=0)

            dout = cp.dot(dout, self.params[W_key].T)

            # Tanh 미분 적용
            if i > 1:
                z_curr = self.cache['z' + str(i-1)]
                dout = dout * (1 - z_curr ** 2)

class AdamW:
    def __init__(self, lr=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = cp.zeros_like(val)
                self.v[key] = cp.zeros_like(val)

        self.t += 1

        for key in params.keys():
            if key in grads:
                params[key] -= self.lr * self.weight_decay * params[key]

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)