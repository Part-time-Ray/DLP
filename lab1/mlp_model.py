import numpy as np

# create mlp without using pytorch
class mlp():
    def __init__(self, d_in, loss='mse', lr=0.01, seed=1, optimizer=0): # optimizer is monentum of beta
        np.random.seed(seed)
        self.activation_functions = {
            'sigmoid': (lambda x: 1 / (1 + np.exp(-x)), lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))),
            'tanh': (lambda x: np.tanh(x), lambda x: 1 - x ** 2),
            'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(int)),
            'none': (lambda x: x, lambda x: 1)
        }
        self.loss_functions = {
            'mse': (lambda x, y: 0.5 * np.mean((x - y) ** 2), lambda x, y: (x - y))
        }
        if loss not in self.loss_functions:
            raise ValueError('Loss function not supported')
        self.loss = self.loss_functions[loss]
        self.d_in = d_in
        self.lr = lr
        self.weights = list()
        self.last_gradient = list()
        self.beta = optimizer
        self.d = list([d_in])
        self.cache = list()
        self.history = list()
    def add_layer(self, nurons, activation = 'none'):
        if activation not in self.activation_functions:
            raise ValueError('Activation function not supported')
        activation = self.activation_functions[activation]
        self.weights.append([np.random.randn(nurons, self.d[-1]), np.random.randn(nurons, 1), activation])
        self.last_gradient.append([np.zeros((nurons, self.d[-1])), np.zeros((nurons, 1))])
        self.d.append(nurons)
    
    def forward(self, X):
        self.cache = list()
        for layer in self.weights:
            x = X.copy()
            X = np.matmul(layer[0], X) + layer[1]
            z = X.copy()
            X = layer[2][0](X)
            self.cache.append((x, z, X))
        return X
    
    def backward(self, x, y, output):
        batch_size = x.shape[1]
        gradient = self.loss[1](output, y) # dL/dy
        for i in range(len(self.weights) - 1, -1, -1):
            W, B, A = self.weights[i]
            last_gradient_w, last_gradient_b = self.last_gradient[i]
            x, z, X = self.cache[i]
            gradient = gradient * A[1](X) # dL/dy * dy/dz,  z = W * f(...) + b
            # dL/db = dL/dy * dy/dz * dz/db, dz/db = 1
            last_gradient_b =  self.beta * last_gradient_b + (1 - self.beta) * np.mean(gradient, axis=1, keepdims=True)
            B -= self.lr * last_gradient_b
            # dL/dw = dL/dy * dy/dz * dz/dw, dz/dw = x
            last_gradient_w = self.beta * last_gradient_w + (1 - self.beta) * (np.matmul(gradient, x.T) / batch_size)
            W -= self.lr * last_gradient_w

            self.last_gradient[i] = [last_gradient_w, last_gradient_b]
            # dL/dx = dL/dy * dy/dz * dz/d(f), dz/d(f) = W
            gradient = np.matmul(W.T, gradient)

    def fit(self, X, Y, epochs=100, batch_size=100):
        X = X.T
        Y = Y.reshape(1, -1)
        num_samples = X.shape[1]
        self.history = list()
        for epoch in range(epochs):
            loss = 0
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_x = X[:, start:end]
                batch_y = Y[:, start:end]

                output = self.forward(batch_x)
                self.backward(batch_x, batch_y, output)

                loss += self.loss[0](output, batch_y) * (end - start)
            loss /= num_samples
            self.history.append(loss) # save loss for each epoch
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}', end = ('\r' if epoch != epochs - 1 else '\n'))
    def test(self, X, Y):
        X = X.T
        Y = Y.reshape(1, -1)
        num_samples = X.shape[1]
        output = self.forward(X)
        loss = self.loss[0](output, Y)
        accuracy = np.sum((output > 0.5) == Y)
        print(f'Loss: {loss}, Accuracy: {accuracy / num_samples:.4f}')
        return (output>0.5).astype(int).reshape(-1)
        
