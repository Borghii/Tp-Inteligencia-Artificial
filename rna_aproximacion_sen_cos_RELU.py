"""
Trabajo práctico: RNA que aproxima f(x,y) = sen(x) + cos(y)

Versión revisada según el apunte del profesor sobre funciones de activación:
- Problema: regresión / aproximación de una función matemática.
- Capas ocultas: ReLU, recomendada para capas ocultas en redes feedforward.
- Capa de salida: lineal, porque la salida es un valor continuo sin restricción.
- Función de pérdida: MSE.

No se usan frameworks profesionales de IA como TensorFlow, Keras, PyTorch o scikit-learn.
Solo se usan numpy para cálculos y matplotlib para gráficos.

Ejecutar:
    python rna_aproximacion_sen_cos_RELU.py

Genera gráficos en:
    figuras_rna_relu/
"""

import os
import copy
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FUNCIÓN OBJETIVO Y DATOS
# ============================================================
def f_objetivo(x, y):
    """f(x,y) = sen(x) + cos(y)."""
    return np.sin(x) + np.cos(y)


def escalar_entrada(xy):
    """Convierte x,y desde [-pi, pi] a [-1, 1]."""
    return xy / np.pi


def crear_grid(n=35):
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi, np.pi, n)
    X, Y = np.meshgrid(x, y)
    XY = np.c_[X.ravel(), Y.ravel()]
    Z = f_objetivo(XY[:, 0], XY[:, 1]).reshape(-1, 1)
    return X, Y, XY, Z


def dividir_train_test(XY, Z, proporcion_train=0.75, seed=1):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(XY))
    corte = int(proporcion_train * len(XY))
    train_idx = indices[:corte]
    test_idx = indices[corte:]
    return XY[train_idx], Z[train_idx], XY[test_idx], Z[test_idx]


# ============================================================
# CAPA Y RED NEURONAL DESDE CERO
# ============================================================
class Layer:
    def __init__(self, num_inputs, num_neurons, activation="relu"):
        self.activation = activation

        # Inicialización según activación.
        # Para ReLU se usa una variante de inicialización He.
        if activation == "relu":
            scale = np.sqrt(2 / num_inputs)
            self.weights = np.random.randn(num_inputs, num_neurons) * scale
        else:
            # Para salida lineal se usa una escala moderada.
            scale = np.sqrt(1 / num_inputs)
            self.weights = np.random.randn(num_inputs, num_neurons) * scale

        self.bias = np.zeros((1, num_neurons))
        self.inputs = None
        self.z = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias

        if self.activation == "relu":
            self.output = np.maximum(0, self.z)
        elif self.activation == "linear":
            self.output = self.z
        else:
            raise ValueError("Activación no soportada")

        return self.output

    def derivative(self):
        if self.activation == "relu":
            return (self.z > 0).astype(float)
        if self.activation == "linear":
            return np.ones_like(self.output)
        raise ValueError("Activación no soportada")

    def backward(self, error, learning_rate):
        """
        error = salida_esperada - salida_predicha.
        Como usamos ese signo, actualizamos sumando el gradiente.
        """
        n = self.inputs.shape[0]
        delta = error * self.derivative()

        old_weights = self.weights.copy()

        self.weights += learning_rate * np.dot(self.inputs.T, delta) / n
        self.bias += learning_rate * np.mean(delta, axis=0, keepdims=True)

        return np.dot(delta, old_weights.T)


class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.03, epochs=1000):
        """
        layers ejemplo: [2, 32, 32, 1]
        Entrada: 2 neuronas para x,y.
        Capas ocultas: ReLU.
        Salida: lineal, porque se predice un valor continuo.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = []
        self.mse_history = []
        self.val_history = []

        for i in range(len(layers) - 1):
            activation = "linear" if i == len(layers) - 2 else "relu"
            self.layers.append(Layer(layers[i], layers[i + 1], activation=activation))

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, inputs, outputs, inputs_val=None, outputs_val=None, verbose=True):
        for epoch in range(self.epochs):
            pred = self.forward(inputs)
            error = outputs - pred
            mse = np.mean(error ** 2)
            self.mse_history.append(float(mse))

            back_error = error
            for layer in reversed(self.layers):
                back_error = layer.backward(back_error, self.learning_rate)

            if inputs_val is not None and outputs_val is not None:
                val_mse = self.mse(inputs_val, outputs_val)
                self.val_history.append(float(val_mse))

            if verbose and (epoch == 0 or (epoch + 1) % max(1, self.epochs // 5) == 0):
                if inputs_val is None:
                    print(f"Época {epoch + 1:5d}/{self.epochs}, MSE train: {mse:.6f}")
                else:
                    print(
                        f"Época {epoch + 1:5d}/{self.epochs}, "
                        f"MSE train: {mse:.6f}, MSE test: {self.val_history[-1]:.6f}"
                    )

    def predict(self, inputs):
        return self.forward(inputs)

    def mse(self, inputs, outputs):
        pred = self.predict(inputs)
        return float(np.mean((outputs - pred) ** 2))


# ============================================================
# GRÁFICOS
# ============================================================
def graficar_superficie(X, Y, Z, titulo, archivo):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
    ax.set_title(titulo)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=28, azim=-135)
    plt.tight_layout()
    plt.savefig(archivo, dpi=160)
    plt.close()


def graficar_aproximaciones(X, Y, Z_real, predicciones, archivo):
    fig = plt.figure(figsize=(11, 8))
    titulos = ["Función real"] + [f"Aprox. {e} épocas" for e in predicciones.keys()]
    superficies = [Z_real] + list(predicciones.values())

    for i, (titulo, Z) in enumerate(zip(titulos, superficies), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")
        ax.set_title(titulo)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=28, azim=-135)

    plt.tight_layout()
    plt.savefig(archivo, dpi=170)
    plt.close()


def graficar_error_epocas(resultados, archivo):
    epocas = [r["epocas"] for r in resultados]
    train = [r["mse_train"] for r in resultados]
    test = [r["mse_test"] for r in resultados]

    plt.figure(figsize=(8, 5))
    plt.plot(epocas, train, marker="o", label="Train")
    plt.plot(epocas, test, marker="o", label="Test")
    plt.yscale("log")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.title("Error al variar épocas")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=160)
    plt.close()


# ============================================================
# EXPERIMENTOS DEL TRABAJO PRÁCTICO
# ============================================================
def experimento_epocas(fig_dir):
    np.random.seed(7)
    X, Y, XY, Z = crear_grid(n=35)
    X_train, y_train, X_test, y_test = dividir_train_test(escalar_entrada(XY), Z)

    epocas_a_probar = [100, 1000, 5000]
    resultados = []
    predicciones = {}

    for epocas in epocas_a_probar:
        print("\nEntrenando con", epocas, "épocas")
        nn = NeuralNetwork([2, 32, 32, 1], learning_rate=0.10, epochs=epocas)
        nn.train(X_train, y_train, verbose=False)

        mse_train = nn.mse(X_train, y_train)
        mse_test = nn.mse(X_test, y_test)
        resultados.append({"epocas": epocas, "mse_train": mse_train, "mse_test": mse_test})
        print(f"MSE train: {mse_train:.6f} | MSE test: {mse_test:.6f}")

        pred = nn.predict(escalar_entrada(XY)).reshape(X.shape)
        predicciones[epocas] = pred

    Z_real = Z.reshape(X.shape)
    graficar_superficie(
        X, Y, Z_real,
        "f(x,y) = sen(x) + cos(y)",
        os.path.join(fig_dir, "superficie_real.png")
    )
    graficar_aproximaciones(
        X, Y, Z_real, predicciones,
        os.path.join(fig_dir, "aproximaciones_epocas.png")
    )
    graficar_error_epocas(resultados, os.path.join(fig_dir, "error_epocas.png"))
    return resultados


def experimento_learning_rate(fig_dir):
    np.random.seed(10)
    _, _, XY, Z = crear_grid(n=30)
    X_train, y_train, X_test, y_test = dividir_train_test(escalar_entrada(XY), Z, seed=3)

    learning_rates = [0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 0.80]
    mse_train, mse_test = [], []

    for lr in learning_rates:
        nn = NeuralNetwork([2, 32, 32, 1], learning_rate=lr, epochs=2000)
        nn.train(X_train, y_train, verbose=False)
        mse_train.append(nn.mse(X_train, y_train))
        mse_test.append(nn.mse(X_test, y_test))
        print(f"LR={lr:<6} | MSE train={mse_train[-1]:.6f} | MSE test={mse_test[-1]:.6f}")

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, mse_train, marker="o", label="Train")
    plt.plot(learning_rates, mse_test, marker="o", label="Test")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("MSE")
    plt.title("Efecto del learning rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "learning_rate.png"), dpi=160)
    plt.close()
    return learning_rates, mse_train, mse_test


def experimento_overfitting(fig_dir):
    """
    Para que se vea el overfitting se usan pocos datos de entrenamiento y ruido.
    La red tiene capacidad suficiente para ajustarse a esos datos ruidosos.
    """
    np.random.seed(22)
    _, _, XY, Z = crear_grid(n=25)
    X_train, y_train, X_test, y_test = dividir_train_test(
        escalar_entrada(XY), Z, proporcion_train=0.035, seed=5
    )

    rng = np.random.default_rng(123)
    y_train_ruidoso = y_train + rng.normal(0, 0.70, size=y_train.shape)

    nn = NeuralNetwork([2, 64, 64, 1], learning_rate=0.02, epochs=2500)
    nn.train(X_train, y_train_ruidoso, X_test, y_test, verbose=False)

    plt.figure(figsize=(8, 5))
    plt.plot(nn.mse_history, label="Train con ruido")
    plt.plot(nn.val_history, label="Test real")
    plt.yscale("log")
    plt.xlabel("Épocas")
    plt.ylabel("MSE")
    plt.title("Ejemplo de overfitting")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "overfitting.png"), dpi=160)
    plt.close()

    return (
        float(nn.mse_history[-1]),
        float(nn.val_history[-1]),
        float(min(nn.val_history)),
        int(np.argmin(nn.val_history) + 1),
    )


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================
if __name__ == "__main__":
    fig_dir = "figuras_rna_relu"
    os.makedirs(fig_dir, exist_ok=True)

    print("=== RNA para aproximar f(x,y)=sen(x)+cos(y) ===")
    print("Arquitectura usada: [2, 32, 32, 1]")
    print("Capas ocultas: ReLU")
    print("Capa de salida: lineal")
    print("Entradas: x/pi, y/pi")
    print("Pérdida: MSE\n")

    resultados = experimento_epocas(fig_dir)

    print("\nResumen al variar épocas:")
    for r in resultados:
        print(
            f"{r['epocas']:5d} épocas | "
            f"MSE train={r['mse_train']:.6f} | MSE test={r['mse_test']:.6f}"
        )

    print("\n=== Learning rate ===")
    experimento_learning_rate(fig_dir)

    print("\n=== Overfitting ===")
    train_final, test_final, test_min, epoca_min = experimento_overfitting(fig_dir)
    print(f"MSE train final: {train_final:.6f}")
    print(f"MSE test final:  {test_final:.6f}")
    print(f"Mejor MSE test:  {test_min:.6f} en época {epoca_min}")

    print("\nGráficos generados en:", os.path.abspath(fig_dir))
