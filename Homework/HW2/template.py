import numpy as np
import math
import matplotlib.pyplot as plt

# -------- activation functions -------
def relu(z):
    return np.maximum(0, z)

def relu_back(xbar, z):
    return xbar * (z > 0)  # Changed from return xbar > 0

def tanh(z):
    return np.tanh(z)

def tanh_back(xbar, z):
    return xbar * (1 - np.tanh(z) ** 2)  # Element-wise multiplication with gradient

identity = lambda z: z

def identity_back(xbar, z):
    return xbar
# -------------------------------------------


# ---------- initialization -----------
def initialization(nin, nout):
    W = np.random.randn(nin, nout) * np.sqrt(2.0 / nin)
    b = np.zeros((nout, 1))
    return W, b
# -------------------------------------


# -------- loss functions -----------
def mse(yhat, y):
    assert yhat.shape == y.shape
    error = yhat - y
    return np.mean(np.square(error))

def mse_back(yhat, y):
    return 2 * (yhat - y) / yhat.size
# -----------------------------------


# ------------- Layer ------------
class Layer:

    def __init__(self, nin, nout, activation=identity):
        self.W, self.b = initialization(nin, nout)
        self.activation = activation
        if activation == tanh:
            self.activation_back = tanh_back
        if activation == relu:
            self.activation_back = relu_back
        if activation == identity:
            self.activation_back = identity_back

        self.cache = {}

    def forward(self, X, train=True):
        Z = self.W.T @ X + self.b
        Xnext = self.activation(Z)
        # save cache
        if train:
            self.cache['Z'] = Z
            self.cache['X_in'] = X  # Store input X
            self.cache['X'] = X  # For backward compatibility
        return Xnext

    def backward(self, XNewBar):
        Z = self.cache['Z']
        X_in = self.cache['X_in']  # Get input X

        print(f"\nDebug shapes in backward:")
        print(f"XNewBar shape: {XNewBar.shape}")
        print(f"Z shape: {Z.shape}")
        print(f"X_in shape: {X_in.shape}")
        print(f"W shape: {self.W.shape}")

        Zbar = self.activation_back(XNewBar, Z)
        print(f"Zbar shape: {Zbar.shape}")
        
        # Calculate gradients
        self.cache['dW'] = X_in @ Zbar.T  # Use input X for weight gradients
        self.cache['db'] = np.sum(Zbar, axis=1, keepdims=True)
        
        # Compute gradient with respect to input
        Xbar = self.W @ Zbar
        print(f"Xbar shape: {Xbar.shape}\n")
        
        return Xbar


class Network:

    def __init__(self, layers, loss):
        self.layers = [*layers]
        self.loss = loss
        self.loss_back = mse_back
        self.cache = {}

    def forward(self, X, y, train=True):

        for layer in self.layers:
            X = layer.forward(X, train)

        yHat = X
        L = self.loss(yHat, y)

        # save cache
        if train:
            self.cache['yHat'] = yHat
            self.cache['L'] = L
            self.cache['y'] = y

        return L, yHat

    def backward(self):
        lBar = self.loss_back(self.cache['yHat'], self.cache['y'])
        xBar = lBar
        for layer in reversed(self.layers):
            xBar = layer.backward(xBar)


class GradientDescent:

    def __init__(self, alpha):
        self.alpha = alpha

    def step(self, network):
        for layer in network.layers:
            layer.W -= self.alpha * layer.cache['dW']
            layer.b -= self.alpha * layer.cache['db']


if __name__ == '__main__':

    # ---------- data preparation ----------------
    # Initialize lists for the numeric data and the string data
    numeric_data = []

    # Read the text file
    with open('auto-mpg.data', 'r') as file:
        for line in file:
            # Split the line into columns
            columns = line.strip().split()

            # Check if any of the first 8 columns contain '?'
            if '?' in columns[:8]:
                continue  # Skip this line if there's a missing value

            # Convert the first 8 columns to floats and append to numeric_data
            numeric_data.append([float(value) for value in columns[:8]])

    # Convert numeric_data to a numpy array for easier manipulation
    numeric_array = np.array(numeric_data)

    # Shuffle the numeric array and the corresponding string array
    nrows = numeric_array.shape[0]
    indices = np.arange(nrows)
    np.random.shuffle(indices)
    shuffled_numeric_array = numeric_array[indices]

    # Split into training (80%) and test (20%) sets
    split_index = int(0.8 * nrows)

    train_numeric = shuffled_numeric_array[:split_index]
    test_numeric = shuffled_numeric_array[split_index:]

    # separate inputs/outputs
    Xtrain = train_numeric[:, 1:]
    ytrain = train_numeric[:, 0]

    Xtest = test_numeric[:, 1:]
    ytest = test_numeric[:, 0]

    # normalize
    Xmean = np.mean(Xtrain, axis=0)
    Xstd = np.std(Xtrain, axis=0)
    ymean = np.mean(ytrain)
    ystd = np.std(ytrain)

    Xtrain = (Xtrain - Xmean) / Xstd
    Xtest = (Xtest - Xmean) / Xstd
    ytrain = (ytrain - ymean) / ystd
    ytest = (ytest - ymean) / ystd

    # reshape arrays (opposite order of pytorch, here we have nx x ns).
    # I found that to be more conveient with the way I did the math operations, but feel free to setup
    # however you like.
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    ytrain = np.reshape(ytrain, (1, len(ytrain)))
    ytest = np.reshape(ytest, (1, len(ytest)))

    # ------------------------------------------------------------

    l1 = Layer(7, 32, relu)
    l2 = Layer(32, 16, relu)
    l3 = Layer(16, 1)

    layers = [l1, l2, l3]
    network = Network(layers, mse)
    alpha = 0.05
    optimizer = GradientDescent(alpha)

    # In the main training loop:
    train_losses = []
    test_losses = []
    epochs = 1500

    for i in range(epochs):
        # Forward pass and backprop
        train_loss, y_train_hat = network.forward(Xtrain, ytrain, train=True)
        network.backward()
        optimizer.step(network)
        
        # Store training loss
        train_losses.append(train_loss)
        
        # Compute and store test loss
        test_loss, y_test_hat = network.forward(Xtest, ytest, train=False)
        test_losses.append(test_loss)
        
        print(f"Epoch {i+1}/{epochs}: Training Loss = {train_loss:.4f}")
        print(f"Epoch {i+1}/{epochs}: Testing Loss = {test_loss:.4f}")



    # --- inference ----
    _, yhat = network.forward(Xtest, ytest, train=False)

    # unnormalize
    yhat = (yhat * ystd) + ymean
    ytest = (ytest * ystd) + ymean

    plt.figure()
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Losses')
    plt.legend()


    plt.figure()
    plt.plot(ytest.T, yhat.T, "o")
    plt.plot([10, 45], [10, 45], "--")

    print("avg error (mpg) =", np.mean(np.abs(yhat - ytest)))

    plt.show()