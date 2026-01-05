import cupy as cp

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1 # 모멘텀 계수
        self.beta2 = beta2 # RMSProp 계수
        self.epsilon = epsilon # 수치 안정성
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
            params[key] -= self.lr * self.weight_decay * params[key]  #가중치 감쇠

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key]**2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t) # 편향 보정
            v_hat = self.v[key] / (1 - self.beta2 ** self.t) # 편향 보정

            params[key] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)