# Import dependencies and write unit tests below
import numpy as np
import pytest
from nn.nn import NeuralNetwork
from nn.preprocess import sample_seqs, one_hot_encode_seqs

def test_single_forward():
    # Validate the `_single_forward` helper for two activations:
    # 1) sigmoid
    # 2) relu
    # Use small arrays to compute the expected outputs explicitly
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="mse")

    # Define W, b, and A_prev directly in the *internal* shape convention:
    # W: (n_out, n_in)
    # b: (n_out, 1)
    # A_prev: (n_in, m)
    W = np.array([[1.0, -1.0]])   # (1,2)
    b = np.array([[0.0]])         # (1,1)
    A_prev = np.array([[1.0, 2.0], [0.0, -1.0]])  # (2,2)

    # Run forward step with sigmoid nonlinearity
    A_curr, Z_curr = nn._single_forward(W, b, A_prev, activation="sigmoid")

    # Manual computation of the expected linear term Z and sigmoid activation A
    Z_expected = np.dot(W, A_prev) + b
    A_expected = 1.0 / (1.0 + np.exp(-Z_expected))

    # Shape check both Z and A --> should be (n_out, m) = (1, 2)
    assert Z_curr.shape == Z_expected.shape
    assert A_curr.shape == A_expected.shape

    # Value check --> compare helper output to manual output
    assert np.allclose(np.array(Z_curr), np.array(Z_expected), atol=1e-6, rtol=0)
    assert np.allclose(np.array(A_curr), np.array(A_expected), atol=1e-6, rtol=0)

    # Repeat with ReLU activation to ensure the activation branch is correct
    W2 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b2 = np.zeros((2, 1))
    A_prev2 = np.array([[-1.0], [2.5]])  # (2,1)
    A_curr2, Z_curr2 = nn._single_forward(W2, b2, A_prev2, activation="relu")

    # Manual expected values
    Z_expected2 = np.dot(W2, A_prev2) + b2
    A_expected2 = np.maximum(0, Z_expected2)

    # Compare
    assert np.allclose(np.array(Z_curr2), np.array(Z_expected2), atol=1e-6, rtol=0)
    assert np.allclose(np.array(A_curr2), np.array(A_expected2), atol=1e-6, rtol=0)

def test_forward():
    # Validate the full `forward()` method on a 1-layer network
    # This checks output values and cache contents (A0, Z1, A1)
    arch = [{'input_dim': 2, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch=arch, lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="bce")

    # Override parameters so the test is deterministic (no randomness)
    nn._param_dict['W1'] = np.array([[2.0, -1.0]])
    nn._param_dict['b1'] = np.array([[-0.5]])

    # Two examples, two features: shape (m, features)
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0]])  # shape (2,2) = (m, features)
    out, cache = nn.forward(X)

    # Manual compute using the same operations as forward():
    # forward() transposes X to get A0 = X.T.
    A_prev = X.T  # (features, m)
    Z_manual = np.dot(nn._param_dict['W1'], A_prev) + nn._param_dict['b1']
    A_manual = 1.0 / (1.0 + np.exp(-Z_manual))

    # Output shape should be (m, n_out) = (2, 1) because forward() transposes back
    assert out.shape == (2, 1)
    assert np.allclose(np.array(out), np.array(A_manual.T), atol=1e-6, rtol=0)

    # Cache check -> forward() should save the input activation A0, plus Z1 and A1
    assert 'A0' in cache and 'A1' in cache and 'Z1' in cache
    # And cached A1 should equal our manually computed activation
    assert np.allclose(np.array(cache['A1']), np.array(A_manual), atol=1e-6, rtol=0)


def test_single_backprop():
    # Validate `_single_backprop` for two activation functions
    # Compare computed gradients (dW, db, dA_prev) to the explicit formulas
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="bce")

    # SIGMOID CASE
    W = np.array([[1.0, -2.0]])      # (1,2)
    b = np.zeros((1,1))
    A_prev = np.array([[1.0, 0.0], [0.0, 1.0]])  # (2,2)

    # Forward linear term
    Z = np.dot(W, A_prev) + b

    # Upstream gradient dA (same shape as Z: (n_out, m))
    dA_curr = np.array([[0.1, -0.2]])
    dA_prev, dW, db = nn._single_backprop(W, b, Z, A_prev, dA_curr, activation_curr='sigmoid')

    # Manual derivatives for sigmoid: dZ = dA * σ(Z)(1-σ(Z))
    s = 1.0 / (1.0 + np.exp(-Z))
    dZ_expected = dA_curr * s * (1 - s)
    m = A_prev.shape[1]

    # Manual affine gradients
    dW_expected = (1.0 / m) * np.dot(dZ_expected, A_prev.T)
    db_expected = (1.0 / m) * np.sum(dZ_expected, axis=1, keepdims=True)
    dA_prev_expected = np.dot(W.T, dZ_expected)

    # Compare to helper results
    assert np.allclose(np.array(dW), np.array(dW_expected), atol=1e-6, rtol=0)
    assert np.allclose(np.array(db), np.array(db_expected), atol=1e-6, rtol=0)
    assert np.allclose(np.array(dA_prev), np.array(dA_prev_expected), atol=1e-6, rtol=0)

    # RELU CASE
    nn2 = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="mse")
    W2 = np.array([[1.0, 0.5]])
    b2 = np.zeros((1,1))
    A_prev2 = np.array([[1.0, -1.0], [0.0, 2.0]])
    Z2 = np.dot(W2, A_prev2) + b2
    dA_curr2 = np.array([[1.0, 2.0]])
    dA_prev2, dW2, db2 = nn2._single_backprop(W2, b2, Z2, A_prev2, dA_curr2, activation_curr='relu')

    # Manual ReLU derivative: pass gradient through only where Z2>0
    dZ_expected2 = dA_curr2.copy()
    dZ_expected2[Z2 <= 0] = 0
    m2 = A_prev2.shape[1]

    # Manual affine gradients
    dW_expected2 = (1.0 / m2) * np.dot(dZ_expected2, A_prev2.T)
    db_expected2 = (1.0 / m2) * np.sum(dZ_expected2, axis=1, keepdims=True)
    dA_prev_expected2 = np.dot(W2.T, dZ_expected2)

    # Compare
    assert np.allclose(np.array(dW2), np.array(dW_expected2), atol=1e-6, rtol=0)
    assert np.allclose(np.array(db2), np.array(db_expected2), atol=1e-6, rtol=0)
    assert np.allclose(np.array(dA_prev2), np.array(dA_prev_expected2), atol=1e-6, rtol=0)

def test_predict():
    # predict() should be a thin wrapper around forward()
    arch = [{'input_dim': 3, 'output_dim': 1, 'activation': 'sigmoid'}]
    nn = NeuralNetwork(nn_arch=arch, lr=0.1, seed=0, batch_size=1, epochs=1, loss_function="bce")
    # set deterministic params
    nn._param_dict['W1'] = np.array([[1.0, 0.0, -1.0]])
    nn._param_dict['b1'] = np.array([[0.0]])
    X = np.array([[1.0, 2.0, 3.0]])
    pred = nn.predict(X)
    out_forward, _ = nn.forward(X)
    assert pred.shape == out_forward.shape
    assert np.allclose(np.array(pred), np.array(out_forward), atol=1e-6, rtol=0)

def test_binary_cross_entropy():
    # Validate BCE loss against the closed-form expression
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    y = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.9999999], [1e-7], [0.5]])

    # Clip for numerical stability just like the implementation
    eps = 1e-15
    yhat_clipped = np.clip(y_hat, eps, 1 - eps)
    m = y.shape[0]
    expected_loss = - (1.0 / m) * np.sum(y * np.log(yhat_clipped) + (1 - y) * np.log(1 - yhat_clipped))
    loss = nn._binary_cross_entropy(y, y_hat)
    # compare with small tolerance
    assert pytest.approx(expected_loss, rel=0, abs=1e-12) == loss

def test_binary_cross_entropy_backprop():
    # Validate BCE gradient w.r.t. predictions
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function='binary_cross_entropy')
    y = np.array([[1.0], [0.0], [1.0]])
    y_hat = np.array([[0.9], [0.2], [0.5]])

    # Implementation expects (n_out, m), so transpose the (m, 1) vectors
    y_T = y.T
    yhat_T = y_hat.T
    dA = nn._binary_cross_entropy_backprop(y_T, yhat_T)

    # Manual derivative
    eps = 1e-15
    yhat_clipped_T = np.clip(yhat_T, eps, 1 - eps)
    m = y.shape[0]
    expected_dA = - (1.0 / m) * (y_T / yhat_clipped_T - (1 - y_T) / (1 - yhat_clipped_T))
    assert np.allclose(np.array(dA), np.array(expected_dA), atol=1e-6, rtol=0)

def test_mean_squared_error():
    # Validate MSE loss value
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function='mean_squared_error')
    y = np.array([[1.0], [2.0]])
    y_hat = np.array([[1.5], [1.0]])
    m = y.shape[0]
    expected_loss = (1.0 / (2 * m)) * np.sum((y_hat - y) ** 2)
    loss = nn._mean_squared_error(y, y_hat)
    assert pytest.approx(expected_loss, rel=0, abs=1e-12) == loss

def test_mean_squared_error_backprop():
    # Validate MSE gradient with respect to predictions
    nn = NeuralNetwork(nn_arch=[], lr=0.1, seed=0, batch_size=1, epochs=1, loss_function='mean_squared_error')
    y = np.array([[1.0], [2.0]])
    y_hat = np.array([[1.5], [1.0]])

    # Implementation expects (n_out, m)
    y_T = y.T
    yhat_T = y_hat.T
    dA = nn._mean_squared_error_backprop(y_T, yhat_T)
    m = y.shape[0]
    expected_dA = (1.0 / m) * (yhat_T - y_T)
    assert np.allclose(np.array(dA), np.array(expected_dA), atol=1e-6, rtol=0)


def test_sample_seqs():
    # Validate that sample_seqs balances classes by upsampling the minority
    # Create tiny dataset with imbalance: many more negatives than positives
    pos = ["AAAA", "TTTT"]
    neg = ["CCCC"] * 5
    seqs = pos + neg
    labels = [True] * len(pos) + [False] * len(neg)

    sampled_seqs, sampled_labels = sample_seqs(seqs, labels)

    # Balanced counts should be equal
    n_pos = sum(1 for lab in sampled_labels if lab)
    n_neg = sum(1 for lab in sampled_labels if not lab)
    assert n_pos == n_neg

    # Ensure total length is as expected: 2 * max(class sizes)
    expected_total = 2 * max(len(pos), len(neg))
    assert len(sampled_seqs) == expected_total

def test_one_hot_encode_seqs():
    # Validate one-hot mapping for single-character sequences plus an unknown base
    seqs = ["A", "T", "C", "G", "N"]
    enc = one_hot_encode_seqs(seqs)

    # Since max_len=1, encoding has 4 features per sequence => shape (5, 4)
    assert enc.shape == (5, 4)
    np.testing.assert_array_equal(enc[0], np.array([1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(enc[1], np.array([0, 1, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(enc[2], np.array([0, 0, 1, 0], dtype=np.float32))
    np.testing.assert_array_equal(enc[3], np.array([0, 0, 0, 1], dtype=np.float32))

    # 'N' is unknown so its encoding should be all zeros
    np.testing.assert_array_equal(enc[4], np.array([0, 0, 0, 0], dtype=np.float32))

    # Multi-length check: AGA has length 3 => 3*4 = 12 features
    enc2 = one_hot_encode_seqs(["AGA"])
    assert enc2.shape == (1, 12)
    expected = np.array([1,0,0,0, 0,0,0,1, 1,0,0,0], dtype=np.float32)
    np.testing.assert_allclose(enc2[0], expected, atol=1e-7)