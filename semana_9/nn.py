from typing import Callable, List
import numpy as np
import torch
from sklearn.metrics import mean_squared_error

def tanh(x) -> np.ndarray:
    return np.tanh(x)

def tanh_prime(x) -> np.ndarray:
    return 1-np.tanh(x)**2

def mse(y_true, y_pred) -> np.float64 | np.ndarray:
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred) -> np.float64 | np.ndarray:
    return 2*(y_pred-y_true)/y_true.size

class Layer:
    def __init__(self):
        self.input: np.ndarray | None = None
        self.output: np.ndarray | None = None

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward_propagation(self, output_error: np.ndarray, learning_rate: np.float64):
        raise NotImplementedError

class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights: np.ndarray = np.random.rand(input_size, output_size) - 0.5
        self.bias: np.ndarray = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: np.float64) -> np.ndarray:
        assert(isinstance(self.input, np.ndarray))

        input_error: np.ndarray = np.dot(output_error, self.weights.T)
        weights_error: np.ndarray = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable):
        self.activation: Callable = activation
        self.activation_prime: Callable = activation_prime

    def forward_propagation(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(input_data)
        return self.output

    def backward_propagation(self, output_error: np.ndarray, learning_rate: np.float64):
        return self.activation_prime(self.input) * output_error

class Network:
    def __init__(self):
        self.layers: List[Layer] = []
        self.loss: Callable | None = None
        self.loss_prime: Callable | None = None

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def use(self, loss: Callable, loss_prime: Callable) -> None:
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data: np.ndarray) -> List[np.ndarray]:
        samples: int = len(input_data)
        result: List[np.ndarray] = []

        output = np.ndarray = np.array([])

        for i in range(samples):
            output = input_data[i]

            for layer in self.layers:
                output = layer.forward_propagation(output)

            result.append(output)

        return result

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, learning_rate: np.float64) -> None:
        assert(isinstance(self.loss, Callable))
        assert(isinstance(self.loss_prime, Callable))
        samples: int = len(x_train)

        for i in range(epochs):
            err: float = 0
            for j in range(samples):
                output: np.ndarray = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss(y_train[j], output)

                error: np.ndarray = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print(f"epoch {i+1}/{epochs} error={err}")



# Simple XOR with Neural Network
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)

net.fit(x_train, y_train, epochs=1000, learning_rate=np.float64(0.1))

out = net.predict(x_train)
print(out)


class NeuralNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = torch.nn.Linear(2, 3)
        self.l2 = torch.nn.Linear(3, 1)

    def forward(self, x):
        x = self.l1(x)
        x = torch.tanh(x)

        x = self.l2(x)
        x = torch.tanh(x)

        return x

net = NeuralNet()
optim = torch.optim.SGD(net.parameters(), lr=0.1)
loss_function = torch.nn.MSELoss()

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

for i in range(1000):
    optim.zero_grad()
    out = net(x_train)
    loss = loss_function(out, y_train)
    loss.backward()
    optim.step()

out = net(x_train)
print(out)
