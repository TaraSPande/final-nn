# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # AFFINE/LINEAR STEP
        # Z = W·A_prev + b
        # W_curr: (n_out, n_in)
        # A_prev: (n_in, m)
        # b_curr: (n_out, 1) broadcast across the m examples
        # Z_curr: (n_out, m)
        Z_curr = np.dot(W_curr, A_prev) + b_curr  # (output_dim, m)

        # NONLINEARITY
        # Apply the layer’s activation elementwise to Z
        if activation.lower() == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation.lower() in ('relu'):
            A_curr = self._relu(Z_curr)
        else:
            # Explicit failure helps debugging if architecture strings are wrong
            raise ValueError(f"Unsupported activation: {activation}")

        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Public inputs come in as X shaped (m, n_features)
        # Internally use A shaped (n_features, m) so that W·A is valid
        A_prev = X.T

        # cache stores intermediates from *this* forward pass
        # Store both Z^l and A^l for each layer l
        # A0 is defined as the input activation (the transposed X)
        cache: Dict[str, ArrayLike] = {}
        cache['A0'] = A_prev

        # Forward propagate through each layer in order
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            # Pull the current layer’s parameters out of the parameter dict
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]

            # Activation choice is stored in the architecture spec
            activation = layer['activation']

            # Run the affine transform + nonlinearity for this layer
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, activation)

            # Save for backprop:
            # Z^l is needed to compute dZ^l from dA^l
            # A^l is needed as the next layer’s A_prev, and also for debugging
            cache['Z' + str(layer_idx)] = Z_curr
            cache['A' + str(layer_idx)] = A_curr

            # Current activation becomes the next layer’s input activation
            A_prev = A_curr

        # Output of last layer is A^L with shape (n_out, m)
        # Transpose back to the public-facing shape (m, n_out)
        output = A_prev.T
        return output, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # m = number of examples in the current mini-batch
        # A_prev is shaped (n_in, m)
        m = A_prev.shape[1]  # number of examples in batch

        # Convert dA (gradient wrt activation output) into dZ (gradient wrt pre-activation) using the derivative of the activation function
        if activation_curr.lower() == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        elif activation_curr.lower() in ('relu'):
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        else:
            raise ValueError(f"Unsupported activation: {activation_curr}")

        # PARAMETER GRADIENTS
        # Given Z = W·A_prev + b:
        # dW = (1/m) · dZ · A_prev^T
        # db = (1/m) · sum(dZ, axis=1)
        # dA_prev = W^T · dZ
        dW_curr = (1.0 / m) * np.dot(dZ_curr, A_prev.T)  # (output_dim, input_dim)
        db_curr = (1.0 / m) * np.sum(dZ_curr, axis=1, keepdims=True)  # (output_dim, 1)
        dA_prev = np.dot(W_curr.T, dZ_curr)  # (input_dim, m)

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # The forward() method outputs y_hat shaped (m, n_out)
        # Many backprop formulas are cleanest using (n_out, m), so this function normalizes shapes and transposes accordingly
        y_arr = np.array(y)
        y_hat_arr = np.array(y_hat)

        # If labels/predictions were passed as vectors, reshape to (m, 1)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if y_hat_arr.ndim == 1:
            y_hat_arr = y_hat_arr.reshape(-1, 1)

        # Convert to internal convention (n_out, m)
        y_T = y_arr.T
        yhat_T = y_hat_arr.T

        # grad_dict will hold gradients for each layer: dWl, dbl for l = 1..L
        grad_dict: Dict[str, ArrayLike] = {}

        # --- START CHAIN RULE
        # Choose the loss derivative based on the configured loss function
        loss_name = self._loss_func.lower()
        if loss_name in ('binary_cross_entropy', 'bce', 'binary_crossentropy'):
            dA_prev = self._binary_cross_entropy_backprop(y_T, yhat_T)
        elif loss_name in ('mean_squared_error', 'mse', 'mean_squared'):
            dA_prev = self._mean_squared_error_backprop(y_T, yhat_T)
        else:
            raise ValueError(f"Unsupported loss function: {self._loss_func}")

        # --- BACKPROPAGATE LAYER L TO LAYER 1
        # At each step, compute gradients for the current layer and propagate dA backwards to the previous layer
        L = len(self.arch)
        for layer_idx in range(L, 0, -1):
            # layer_idx is 1-based here (1..L) to match parameter naming 'W1', 'b1', ...
            # arch list is 0-based so arch[layer_idx-1] is the current layer spec
            activation_curr = self.arch[layer_idx - 1]['activation']

            # A^{l-1} and Z^l were cached during forward pass
            A_prev = cache['A' + str(layer_idx - 1)]  # (input_dim, m)
            Z_curr = cache['Z' + str(layer_idx)]      # (output_dim, m)

            # Current parameters
            W_curr = self._param_dict['W' + str(layer_idx)]
            b_curr = self._param_dict['b' + str(layer_idx)]

            # Compute dA^{l-1}, dW^l, db^l
            dA_prev, dW_curr, db_curr = self._single_backprop(
                W_curr=W_curr,
                b_curr=b_curr,
                Z_curr=Z_curr,
                A_prev=A_prev,
                dA_curr=dA_prev,
                activation_curr=activation_curr
            )

            # Store layer gradients using the same 1-based suffix convention
            grad_dict['dW' + str(layer_idx)] = dW_curr
            grad_dict['db' + str(layer_idx)] = db_curr

            # Prepare for next iteration (moving to previous layer): dA_prev already refers to dA^{l-1} at this point.
            dA_prev = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # L = number of layers; update each layer’s W and b with gradient descent
        L = len(self.arch)
        for layer_idx in range(1, L + 1):
            # Gradient descent:
            # W := W - α dW
            # b := b - α db
            self._param_dict['W' + str(layer_idx)] -= self._lr * grad_dict['dW' + str(layer_idx)]
            self._param_dict['b' + str(layer_idx)] -= self._lr * grad_dict['db' + str(layer_idx)]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Convert inputs to NumPy arrays to reliably index/slice and do math
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        # m = number of training examples
        m = X_train.shape[0]

        # Avoid having a batch size larger than the dataset
        batch_size = min(self._batch_size, m)

        # Track average loss per epoch for both training and validation
        per_epoch_loss_train: List[float] = []
        per_epoch_loss_val: List[float] = []

        for epoch in range(self._epochs):
            # Shuffle to start each epoch
            # Makes mini-batches different each epoch and generally improves learning
            perm = np.random.permutation(m)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            # MINI BATCH LOOP
            # Iterate start indices 0, batch_size, 2*batch_size, ...
            epoch_losses = []
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass: compute predictions for this mini-batch
                y_hat_batch, cache = self.forward(X_batch)

                # Compute scalar loss value for this mini-batch
                loss_name = self._loss_func.lower()
                if loss_name in ('binary_cross_entropy', 'bce', 'binary_crossentropy'):
                    loss_val = self._binary_cross_entropy(y_batch, y_hat_batch)
                elif loss_name in ('mean_squared_error', 'mse', 'mean_squared'):
                    loss_val = self._mean_squared_error(y_batch, y_hat_batch)
                else:
                    raise ValueError(f"Unsupported loss function: {self._loss_func}")

                # Keep track of mini-batch loss to average over the epoch
                epoch_losses.append(loss_val)

                # Backprop: compute gradients for each parameter based on this mini-batch
                grad_dict = self.backprop(y_batch, y_hat_batch, cache)

                # Update: apply gradient descent step to parameters
                self._update_params(grad_dict)

            # Record average training loss for this epoch (mean over mini-batches)
            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            per_epoch_loss_train.append(avg_train_loss)

            # Validation loss: run a single forward pass over the full validation set
            # (No parameter updates based on validation)
            y_hat_val, _ = self.forward(X_val)
            if self._loss_func.lower() in ('binary_cross_entropy', 'bce', 'binary_crossentropy'):
                val_loss = self._binary_cross_entropy(y_val, y_hat_val)
            else:
                val_loss = self._mean_squared_error(y_val, y_hat_val)
            per_epoch_loss_val.append(float(val_loss))

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # Prediction is just a forward pass
        y_hat, _ = self.forward(X)
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # σ(z) = 1 / (1 + e^{-z})
        return 1.0 / (1.0 + np.exp(-Z))

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # If A = σ(Z), then dA/dZ = σ(Z)(1-σ(Z))
        # By chain rule: dZ = dA ⊙ dA/dZ
        s = self._sigmoid(Z)
        dZ = dA * s * (1 - s)
        return dZ

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # ReLU(Z) = max(0, Z) elementwise
        return np.maximum(0, Z)

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # ReLU derivative is 1 when Z>0, else 0
        # Copy dA and zero out entries where ReLU was inactive
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Convert to arrays to safely apply vectorized ops
        y = np.array(y)
        y_hat = np.array(y_hat)

        # Ensure shapes are (m, n_out)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)

        # m = number of examples
        m = y.shape[0]

        # Numerical stability: clip predictions away from exactly 0 or 1 so log(.) does not produce -inf
        eps = 1e-15
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)

        # BCE loss: L = -(1/m) Σ [ y log(p) + (1-y) log(1-p) ]
        loss = - (1.0 / m) * np.sum(y * np.log(y_hat_clipped) + (1 - y) * np.log(1 - y_hat_clipped))
        return float(loss)

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Expect y and y_hat to be shaped (n_out, m) here
        y = np.array(y)
        y_hat = np.array(y_hat)

        # Same numerical stability clipping as the forward loss
        eps = 1e-15
        y_hat_clipped = np.clip(y_hat, eps, 1 - eps)

        # m = number of examples (columns)
        m = y.shape[1]  # number of examples

        # Derivative of BCE wrt predictions A (= y_hat): ∂L/∂A = -(1/m) · ( y/A - (1-y)/(1-A) )
        dA = - (1.0 / m) * (y / y_hat_clipped - (1 - y) / (1 - y_hat_clipped))
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        # Convert to arrays 
        y = np.array(y)
        y_hat = np.array(y_hat)

        # Ensure (m, n_out) shape 
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)

        # m = number of examples 
        m = y.shape[0]

        # Use (1/(2m)) factor so derivative is clean: ∂L/∂y_hat = (1/m) (y_hat - y)
        loss = (1.0 / (2 * m)) * np.sum((y_hat - y) ** 2)
        return float(loss)

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Expect y and y_hat shaped (n_out, m)
        y = np.array(y)
        y_hat = np.array(y_hat)

        # m = number of examples
        m = y.shape[1]

        # With our definition of MSE = (1/(2m)) Σ (y_hat - y)^2, derivative wrt y_hat is (1/m) (y_hat - y).
        dA = (1.0 / m) * (y_hat - y)
        return dA