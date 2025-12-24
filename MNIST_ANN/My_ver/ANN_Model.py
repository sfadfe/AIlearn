import cupy as cp

class ANN_class:
    def __init__(self, input_size, hidden_sizes, output_size):

        self.params = {}
        self.hidden_sizes = hidden_sizes
        
        all_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(all_sizes) - 1):
            in_node = all_sizes[i]
            out_node = all_sizes[i+1]
            
            w_key = 'W' + str(i + 1)
            b_key = 'b' + str(i + 1)
            
            scale = cp.sqrt(2.0 / in_node)
            self.params[w_key] = cp.random.randn(in_node, out_node) * scale
            self.params[b_key] = cp.zeros(out_node)
    
        self.num_layers = len(all_sizes) - 1

    def forward(self, x):
        self.cache = {}
        self.cache['z0'] = x

        for i in range(1, self.num_layers + 1):
            
            W = self.params['W' + str(i)]
            b = self.params['b' + str(i)]
            
            x = cp.dot(x, W) + b
            
            if i != self.num_layers:
                x = cp.maximum(0, x)
            
            self.cache['z' + str(i)] = x
            
        return x

    def gradient(self, x, t):
        grads = {}
        batch_size = x.shape[0]
        
        L = self.num_layers
        y = self.cache['z' + str(L)]
        y = self.softmax(y)
        
        delta = (y - t) / batch_size
        z_prev = self.cache['z' + str(L - 1)]
        grads['W' + str(L)] = cp.dot(z_prev.T, delta)
        grads['b' + str(L)] = cp.sum(delta, axis=0)

        for i in range(L - 1, 0, -1):
            W_next = self.params['W' + str(i + 1)]
            delta = cp.dot(delta, W_next.T)
            
            z_curr = self.cache['z' + str(i)]
            delta = delta * (z_curr > 0)
            z_prev = self.cache['z' + str(i - 1)]
            
            grads['W' + str(i)] = cp.dot(z_prev.T, delta)
            grads['b' + str(i)] = cp.sum(delta, axis=0)
            
        return grads
    
    def softmax(self, a):
        c = cp.max(a, axis=1, keepdims=True) # overflow 방지: 가장 큰 값 뺌 
        exp_a = cp.exp(a - c)
        sum_exp_a = cp.sum(exp_a, axis=1, keepdims=True)
        return exp_a / sum_exp_a
    
    def loss(self, x, t, y):
        
        batch_size = x.shape[0]
        delta = 1e-7 # 만약 softmax 값이 0이면 log0이 나와 무한대가 나옴: 아주 작은값을 더해야함.
        
        loss = -cp.sum(t * cp.log(y + delta)) / batch_size 
        return loss
    
    def accuracy(self, x, t, y):

        y_pred = cp.argmax(y, axis=1)
        t_true = cp.argmax(t, axis=1)
        
        accuracy = cp.sum(y_pred == t_true) / x.shape[0]
        return accuracy