import numpy as np


class linear_t:
    def __init__(self, alpha=784, c=10):
        # initialize to appropriate sizes, fill with Gaussian entires
        # normalize to make the Frobenius norm of w, b equal to 1
        self.W = np.random.randn(c, alpha)
        self.b = np.random.randn(c)
        # Normalize w and b
        self.W /= np.linalg.norm(self.W, 'fro')
        self.b /= np.linalg.norm(self.b)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.h_l = None

    def forward(self, h_l):
        h_lp1 = h_l @ self.W.T + self.b
        # cache h_l in forward because we will need it to compute
        # dw in backward
        self.h_l = h_l
        return h_lp1

    def backward(self, dh_lp1):
        dh_l = dh_lp1 @ self.W
        dW = dh_lp1.reshape(-1, 1) @ self.h_l.reshape(1, -1)
        db = np.sum(dh_lp1, axis=0)
        self.dW, self.db = dW, db
        # notice that there is no need to cache dh_l
        return dh_l

    def zero_grad(self):
        # useful to delete the stored backprop gradients of the
        # previous mini -batch before you start a new mini -batch
        self.dW, self.db = 0 * self.dW, 0 * self.db


class relu_t:
    def __init__(self):
        self.h_l = None  # To store the input to the layer

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, h_l):
        self.h_l = h_l
        return self.relu(h_l)

    def backward(self, dh_lp1):
        return dh_lp1 * (self.h_l > 0)


class softmax_cross_entropy_t:
    def __init__(self):
        self.h_lp1 = None  # To store softmax probabilities for use in backward pass
        self.y = None  # To store the true labels

    def forward(self, h_l, y):
        """
        h_l: np.ndarray (batch_size, num_classes), or (num_classes,) for one sample
        y: np.ndarray (batch_size,) or (batch_size, 1)
        """
        if not isinstance(y, np.ndarray):
            y = np.array([y], dtype=int)
        self.y = y
        if len(h_l.shape) == 1:
            h_l = h_l.reshape(1, -1)
        self.h_lp1 = np.exp(h_l) / np.sum(np.exp(h_l), axis=1, keepdims=True)
        if len(self.h_lp1.shape) == 1:
            self.h_lp1 = self.h_lp1.reshape(1, -1)
        # compute average loss ell(y) over a mini-batch
        # need to consider the case when there is only one sample
        # i.e. the shape of y is (1,), self.h_lp1 should be reshaped
        # to (1, c) to avoid broadcasting issues
        ell = -np.sum(np.log(self.h_lp1[np.arange(len(y)), y])) / len(y)
        y = np.asarray(y).reshape(-1)  # Ensure y is a 1D array
        error = np.mean(np.argmax(self.h_lp1, axis=1) != y)
        return self.h_lp1, ell, error

    def backward(self, dh_lp1=None, y=None):
        # as we saw in the notes, the backprop input to the loss layer is 1,
        # so this function does not take any arguments
        if dh_lp1 is not None:
            self.dh_lp1 = dh_lp1
        if y is not None:
            self.y = y
        if not isinstance(self.y, np.ndarray):
            self.y = np.array([self.y], dtype=int)
        dh_l = self.h_lp1.copy()
        if len(dh_l.shape) == 1:
            dh_l = dh_l.reshape(1, -1)
        dh_l[np.arange(len(self.y)), self.y] -= 1
        dh_l /= len(self.y)  # Average over the batch size
        return dh_l


# Compare with PyTorch
if __name__ == "__main__":
    
    import torch
    import torch.nn as nn
    import torch.optim as optim

    # Your numpy-based linear layer test function
    def test_linear():
        # Define input parameters
        alpha = 784  # Input dimension
        c = 10       # Output dimension
        batch_size = 5

        # Create an instance of your class
        linear_np = linear_t(alpha=alpha, c=c)

        # Generate random input
        h_l_np = np.random.randn(batch_size, alpha)

        # Forward pass using your implementation
        out_np = linear_np.forward(h_l_np)

        # Now let's do the same with PyTorch's Linear layer
        linear_torch = nn.Linear(alpha, c)
        
        # Copy weights and biases from numpy to torch
        with torch.no_grad():
            linear_torch.weight.copy_(torch.tensor(linear_np.W))
            linear_torch.bias.copy_(torch.tensor(linear_np.b))

        # Forward pass using PyTorch
        h_l_torch = torch.tensor(h_l_np, dtype=torch.float32)
        out_torch = linear_torch(h_l_torch).detach().numpy()

        # Compare outputs
        print("Output difference (Linear):", np.max(np.abs(out_np - out_torch)))

    # Your numpy-based ReLU test function
    def test_relu():
        relu_np = relu_t()

        # Generate random input
        h_l_np = np.random.randn(5, 10)

        # Forward pass using your ReLU
        out_np = relu_np.forward(h_l_np)

        # Now with PyTorch
        relu_torch = nn.ReLU()

        h_l_torch = torch.tensor(h_l_np, dtype=torch.float32)
        out_torch = relu_torch(h_l_torch).detach().numpy()

        # Compare outputs
        print("Output difference (ReLU):", np.max(np.abs(out_np - out_torch)))

    # Your numpy-based softmax cross-entropy test function
    def test_softmax_cross_entropy():
        # Define input parameters
        batch_size = 5
        num_classes = 10

        softmax_np = softmax_cross_entropy_t()

        # Generate random inputs
        h_l_np = np.random.randn(batch_size, num_classes)
        y_np = np.random.randint(0, num_classes, size=batch_size)

        # Forward pass using your softmax cross entropy
        softmax_out_np, ell_np, error_np = softmax_np.forward(h_l_np, y_np)

        # Now with PyTorch
        loss_fn_torch = nn.CrossEntropyLoss()

        h_l_torch = torch.tensor(h_l_np, dtype=torch.float32)
        y_torch = torch.tensor(y_np, dtype=torch.int64)

        # PyTorch loss
        loss_torch = loss_fn_torch(h_l_torch, y_torch).item()

        # Compare outputs
        print("Output difference (h_lp1):", np.max(np.abs(softmax_out_np - torch.softmax(h_l_torch, dim=1).detach().numpy())))
        print("Loss difference (Softmax Cross Entropy):", np.abs(ell_np - loss_torch))
        print("Error difference (Softmax Cross Entropy):", np.abs(error_np - (np.mean(np.argmax(softmax_out_np, axis=1) != y_np))))

    # Run tests
    test_linear()
    test_relu()
    test_softmax_cross_entropy()
    