import math


class AutoGrad:  ## from Autograd import AutoGrad as AG 로 사용 바람.
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # 초기 노드 미분값은 0으로 초기화함 
        self._backward = lambda: None
        self._prev = set(_children) # 그래프 연결 정보
        self._op = _op # 디버깅용 연산자 표시

    def __add__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)

        out = AutoGrad(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, AutoGrad) else AutoGrad(other)

        out = AutoGrad(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def sin(self):
        out = AutoGrad(math.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += math.cos(self.data) * out.grad
        out._backward = _backward

        return out
    
    def cos(self):
        out = AutoGrad(math.cos(self.data), (self,), 'cos')

        def _backward():
            self.grad += -math.sin(self.data) * out.grad
        out._backward = _backward

        return out
    
    def exp(self):
        out = AutoGrad(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def log(self):
        out = AutoGrad(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, power):
        out = AutoGrad(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()