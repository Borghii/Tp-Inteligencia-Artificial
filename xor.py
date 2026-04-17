import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# NEURONA
# -------------------------
class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(size=num_inputs)
        self.bias = np.random.uniform()

    def activate(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(np.dot(inputs, self.weights) + self.bias)
        return self.output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self):
        return self.output * (1 - self.output)

    def update_weights(self, delta, learning_rate):
        self.weights += learning_rate * delta * self.inputs
        self.bias += learning_rate * delta


# -------------------------
# CAPA
# -------------------------
class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors, learning_rate):
        deltas = []

        for i, neuron in enumerate(self.neurons):
            delta = errors[i] * neuron.sigmoid_derivative()
            neuron.update_weights(delta, learning_rate)
            deltas.append(delta)

        return np.dot(np.array([neuron.weights for neuron in self.neurons]).T, deltas)


# -------------------------
# RED NEURONAL
# -------------------------
class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []

        for i in range(len(layers) - 1):
            self.layers.append(Layer(layers[i+1], layers[i]))

    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            total_error = 0

            for x, y in zip(inputs, outputs):
                # FORWARD
                activations = [x]
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))

                # ERROR
                output_errors = y - activations[-1]
                total_error += np.sum(output_errors ** 2)

                # BACKWARD
                errors = output_errors
                for i in reversed(range(len(self.layers))):
                    errors = self.layers[i].backward(errors, self.learning_rate)

            if epoch % 1000 == 0:
                mse = total_error / len(inputs)
                print(f"Epoch {epoch}, MSE: {mse}")

    def predict(self, inputs):
        activations = inputs
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations


# -------------------------
# DATOS CLIENTES
# -------------------------

# interes - descuento

# -------------------------
# DATOS CLIENTES
# -------------------------

# interes - descuento
inputs = np.array([
    [0.10, 0.10],
    [0.20, 0.80],
    [0.80, 0.20],
    [0.90, 0.90],
    [0.40, 0.40],
    [0.70, 0.60],
    [0.30, 0.70],
    [0.60, 0.30],
    [0.50, 0.50],
    [0.85, 0.15],
    [0.15, 0.85],
    [0.95, 0.95]
])

outputs = np.array([
    [0],
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1]
])

# -------------------------
# ENTRENAMIENTO
# -------------------------
layers = [2, 2, 1] 

nn = NeuralNetwork(layers, learning_rate=0.1, epochs=10000)
nn.train(inputs, outputs)


# -------------------------
# RESULTADOS
# -------------------------
predicted_output = np.array([nn.predict(x) for x in inputs])

print("\nPredicciones:")
print(np.round(predicted_output, 3))

print("\nRedondeado:")
print(np.round(predicted_output).astype(int))

accuracy = np.mean(np.round(predicted_output).ravel() == outputs.ravel())
print("\nAccuracy:", accuracy)





