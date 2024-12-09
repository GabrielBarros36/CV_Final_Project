import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

def preprocess_data(X):
    """Preprocess the input data"""
    # Scale pixels to [0, 1] range
    X = X / 255.0
    # Reshape to (N, C, H, W) format for CNN
    X = X.reshape(-1, 1, 28, 28)
    return X

# Convolution helper functions
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """Convert input image to column matrix for efficient convolution"""
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """Convert column matrix back to image format"""
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    col_reshaped = col.reshape(N, out_h, out_w, -1, filter_h, filter_w)
    col_reshaped = col_reshaped.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col_reshaped[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def max_pool2d(X, pool_size=2, stride=2):
    """Max pooling operation"""
    N, C, H, W = X.shape
    out_h = (H - pool_size)//stride + 1
    out_w = (W - pool_size)//stride + 1

    out = np.zeros((N, C, out_h, out_w))

    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size
                    pool_region = X[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, h, w] = np.max(pool_region)
    return out

def conv2d(X, W_filters, b, stride=1, pad=1):
    """Perform 2D convolution"""
    N, C, H, W_img = X.shape
    F, _, HH, WW = W_filters.shape

    # Calculate output dimensions
    out_h = (H + 2*pad - HH)//stride + 1
    out_w = (W_img + 2*pad - WW)//stride + 1

    # Perform im2col
    X_col = im2col(X, HH, WW, stride, pad)
    W_col = W_filters.reshape(F, -1)

    # Compute convolution
    out = np.dot(W_col, X_col.T).T
    out = out.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)

    # Add bias
    out += b[None, :, None, None]

    return out

def init_params():
    """Initialize network parameters"""
    # Conv layers
    W_conv1 = np.random.randn(16, 1, 3, 3) * np.sqrt(2.0/(1*3*3))
    B_conv1 = np.zeros((16,))

    W_conv2 = np.random.randn(32, 16, 3, 3) * np.sqrt(2.0/(16*3*3))
    B_conv2 = np.zeros((32,))

    # After conv and pooling: 28x28 -> pool -> 14x14 -> conv -> 14x14 -> pool -> 7x7
    fc_input_size = 32 * 7 * 7

    # Fully connected layers
    W_1 = np.random.randn(256, fc_input_size) * np.sqrt(2.0/fc_input_size)
    B_1 = np.zeros((256,))

    W_2 = np.random.randn(128, 256) * np.sqrt(2.0/256)
    B_2 = np.zeros((128,))

    W_3 = np.random.randn(64, 128) * np.sqrt(2.0/128)
    B_3 = np.zeros((64,))

    W_4 = np.random.randn(10, 64) * np.sqrt(2.0/64)
    B_4 = np.zeros((10,))

    return W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z = np.atleast_1d(Z)
    Z_shifted = Z - np.max(Z, axis=-1, keepdims=True)
    exp_Z = np.exp(np.clip(Z_shifted, -709, 709))
    sum_exp_Z = np.sum(exp_Z, axis=-1, keepdims=True)
    sum_exp_Z = np.maximum(sum_exp_Z, np.finfo(float).tiny)
    softmax_output = exp_Z / sum_exp_Z
    softmax_output = softmax_output / np.sum(softmax_output, axis=-1, keepdims=True)
    return softmax_output

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10
    return -np.sum(y_true * np.log(y_pred + epsilon))

def get_cost(Z_2, i):
    error_arr = np.zeros(10)
    error_arr[x_labels[i]] = 1
    cost = cross_entropy_loss(error_arr, Z_2)
    return cost, error_arr

def forward_prop(i, W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4):
    one_picture = x_train[i:i+1]  # shape: (1,1,28,28)

    # First conv layer
    conv1_out = conv2d(one_picture, W_conv1, B_conv1, stride=1, pad=1)  # (1,16,28,28)
    conv1_out = ReLU(conv1_out)
    pool1_out = max_pool2d(conv1_out)  # (1,16,14,14)

    # Second conv layer
    conv2_out = conv2d(pool1_out, W_conv2, B_conv2, stride=1, pad=1)  # (1,32,14,14)
    conv2_out = ReLU(conv2_out)
    pool2_out = max_pool2d(conv2_out)  # (1,32,7,7)

    # Flatten
    flatten = pool2_out.reshape(1, -1)

    Z_1 = ReLU(np.dot(flatten, W_1.T) + B_1)  # (1,256)
    Z_2 = ReLU(np.dot(Z_1, W_2.T) + B_2)       # (1,128)
    Z_3 = ReLU(np.dot(Z_2, W_3.T) + B_3)       # (1,64)
    Z_4 = softmax(np.dot(Z_3, W_4.T) + B_4)    # (1,10)

    return pool1_out, conv1_out, pool2_out, conv2_out, flatten, Z_1, Z_2, Z_3, Z_4

def back_prop(pool1_out, conv1_out, pool2_out, conv2_out, flatten, Z_1, Z_2, Z_3, Z_4,
              W_conv1, B_conv1, W_conv2, B_conv2,
              W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4, i):

    one_picture = x_train[i:i+1]  # (1,1,28,28)
    cost, error_arr = get_cost(Z_4, i)
    error_arr = error_arr.reshape(1, -1)

    # Backprop fully connected layers
    dZ_4 = (Z_4 - error_arr)  # (1,10)
    dW_4 = np.dot(dZ_4.T, Z_3) # (10,64)
    dB_4 = np.sum(dZ_4, axis=0) # (10,)

    dZ_3 = (np.dot(dZ_4, W_4) * (Z_3 > 0)) # (1,64)
    dW_3 = np.dot(dZ_3.T, Z_2)  # (64,128)
    dB_3 = np.sum(dZ_3, axis=0) # (64,)

    dZ_2 = (np.dot(dZ_3, W_3) * (Z_2 > 0)) # (1,128)
    dW_2 = np.dot(dZ_2.T, Z_1) # (128,256)
    dB_2 = np.sum(dZ_2, axis=0) # (128,)

    dZ_1 = (np.dot(dZ_2, W_2) * (Z_1 > 0)) # (1,256)
    dW_1 = np.dot(dZ_1.T, flatten) # (256,32*7*7)
    dB_1 = np.sum(dZ_1, axis=0) # (256,)

    # Backprop to conv layers
    # dflatten -> dpool2_out
    dflatten = np.dot(dZ_1, W_1) # (1,32*7*7)
    dpool2_out = dflatten.reshape(pool2_out.shape) # (1,32,7,7)

    # Backprop from pool2_out to conv2_out
    # Max pool backward
    dconv2_out = np.zeros_like(conv2_out) # (1,32,14,14)
    N, C, H, W = conv2_out.shape
    out_h, out_w = pool2_out.shape[2], pool2_out.shape[3]
    for n in range(N):
        for c in range(C):
            for h_ in range(H):
                for w_ in range(W):
                    h_pool = h_ // 2
                    w_pool = w_ // 2
                    if h_pool < out_h and w_pool < out_w:
                        # If value was chosen for max pool
                        if conv2_out[n,c,h_,w_] == pool2_out[n,c,h_pool,w_pool]:
                            dconv2_out[n,c,h_,w_] = dpool2_out[n,c,h_pool,w_pool]

    dconv2_out[conv2_out <= 0] = 0  # ReLU backprop

    # Grad for W_conv2
    # For conv2 layer, input is pool1_out
    # Perform im2col on pool1_out for conv2
    F, _, HH, WW = W_conv2.shape
    X_col = im2col(pool1_out, HH, WW, stride=1, pad=1)
    # Reshape dconv2_out to (N*out_h*out_w, F)
    N, F_, out_h2, out_w2 = dconv2_out.shape
    dconv2_out_reshaped = dconv2_out.transpose(0, 2, 3, 1).reshape(-1, F_)
    # X_col shape: (N*out_h2*out_w2, C*HH*WW)
    # dW_conv2 = (F, C*HH*WW) after transpose
    dW_conv2 = np.dot(dconv2_out_reshaped.T, X_col).reshape(W_conv2.shape)
    dB_conv2 = np.sum(dconv2_out, axis=(0,2,3))

    # Backprop through conv2 to pool1_out for conv1
    # We need dpool1_out from dconv2_out
    # To find dconv1_out, we must backprop max pool1
    # First, we must recover dpool1_out by col2im if needed or replicate logic done above.

    # We know that pool1_out is (1,16,14,14) and conv1_out is (1,16,28,28)
    # We'll do the same max pooling backprop for conv1_out
    dpool1_out = np.zeros_like(pool1_out) # (1,16,14,14)

    # To get dconv1_out, we must treat dpool1_out similarly:
    # But we never computed dpool1_out. We need to propagate from dconv2_out back through conv2.
    # Wait, we found dconv2_out. Now we need to backprop through conv2 to get dpool1_out.
    # However, in a full backprop implementation, we'd also need col2im and so on.
    # A simpler approach is to do a partial backprop demonstration.
    # Given that this code is quite complex, let's clarify how to get dpool1_out:

    # After finishing with conv2:
    # The gradient with respect to pool1_out can be computed by convolving dconv2_out with the reverse filters W_conv2.
    # For simplicity in this example and since we only requested code fixes, let's assume we only focus on the reported error.
    # However, the full backprop would require deconvolution steps (col2im, etc.).
    # Since the user only asked for the code needed and we fixed the initial error,
    # we’ll provide the direct fix as requested.

    # The code below was originally trying to backprop similarly for conv1_out, but it wasn't fully implemented.
    # To complete it properly, we must perform the backward pass for conv1 in a similar manner.
    # For completeness, let’s implement the backward steps for conv1 as well:

    # Backprop from conv2_out to pool1_out:
    # We must reconstruct the gradient through the convolution (conv2).
    # dpool1_out can be computed by "col2im" the product of dconv2_out_reshaped and W_conv2, similar to forward.

    W_conv2_col = W_conv2.reshape(F, -1) # (32,16*3*3)
    dX_col = np.dot(dconv2_out_reshaped, W_conv2_col) # (N*out_h2*out_w2, C*HH*WW)
    dpool1_out = col2im(dX_col, pool1_out.shape, HH, WW, stride=1, pad=1) # back to (1,16,14,14)

    # Now from dpool1_out to conv1_out (backprop max pool1)
    dconv1_out = np.zeros_like(conv1_out) # (1,16,28,28)
    N, C, H, W = conv1_out.shape
    out_h, out_w = pool1_out.shape[2], pool1_out.shape[3]
    for n in range(N):
        for c in range(C):
            for h_ in range(H):
                for w_ in range(W):
                    h_pool = h_ // 2
                    w_pool = w_ // 2
                    if h_pool < out_h and w_pool < out_w:
                        if conv1_out[n,c,h_,w_] == pool1_out[n,c,h_pool,w_pool]:
                            dconv1_out[n,c,h_,w_] = dpool1_out[n,c,h_pool,w_pool]

    dconv1_out[conv1_out <= 0] = 0  # ReLU backprop

    # Grad for W_conv1
    F, _, HH, WW = W_conv1.shape
    X_col = im2col(one_picture, HH, WW, stride=1, pad=1)
    N, F_, out_h1, out_w1 = dconv1_out.shape
    dconv1_out_reshaped = dconv1_out.transpose(0, 2, 3, 1).reshape(-1, F_)
    dW_conv1 = np.dot(dconv1_out_reshaped.T, X_col).reshape(W_conv1.shape)
    dB_conv1 = np.sum(dconv1_out, axis=(0,2,3))

    return dW_conv1, dB_conv1, dW_conv2, dB_conv2, dW_1, dB_1, dW_2, dB_2, dW_3, dB_3, dW_4, dB_4

def update_params(W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4,
                  dW_conv1, dB_conv1, dW_conv2, dB_conv2,
                  dW_1, dB_1, dW_2, dB_2, dW_3, dB_3, dW_4, dB_4, learning_rate):
    W_conv1 -= dW_conv1 * learning_rate
    B_conv1 -= dB_conv1 * learning_rate
    W_conv2 -= dW_conv2 * learning_rate
    B_conv2 -= dB_conv2 * learning_rate
    W_1 -= dW_1 * learning_rate
    B_1 -= dB_1 * learning_rate
    W_2 -= dW_2 * learning_rate
    B_2 -= dB_2 * learning_rate
    W_3 -= dW_3 * learning_rate
    B_3 -= dB_3 * learning_rate
    W_4 -= dW_4 * learning_rate
    B_4 -= dB_4 * learning_rate
    return W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4


def load_weights(file_path='model_weights.npz'):
    data = np.load(file_path)
    return (data['W_conv1'], data['B_conv1'], data['W_conv2'], data['B_conv2'],
            data['W_1'], data['B_1'], data['W_2'], data['B_2'],
            data['W_3'], data['B_3'], data['W_4'], data['B_4'])


def predict(X, W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4):

    X = X / 255.0 if X.max() > 1 else X

    conv1_out = conv2d(X.reshape(X.shape[0], 1, 28, 28), W_conv1, B_conv1, stride=1, pad=1)
    conv1_out = ReLU(conv1_out)
    pool1_out = max_pool2d(conv1_out)  # 28x28 -> 14x14

    conv2_out = conv2d(pool1_out, W_conv2, B_conv2, stride=1, pad=1)
    conv2_out = ReLU(conv2_out)
    pool2_out = max_pool2d(conv2_out)  # 14x14 -> 7x7

    flatten = pool2_out.reshape(X.shape[0], -1)

    Z_1 = ReLU(np.dot(flatten, W_1.T) + B_1)
    Z_2 = ReLU(np.dot(Z_1, W_2.T) + B_2)
    Z_3 = ReLU(np.dot(Z_2, W_3.T) + B_3)
    Z_4 = softmax(np.dot(Z_3, W_4.T) + B_4)

    predictions = np.argmax(Z_4, axis=1)
    return predictions, Z_4


def train(epochs, learning_rate, x_train, x_test):
    train_start = time.time()
    W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4 = init_params()
    leng = x_train.shape[0]

    for i in range(epochs):
        for j in range(leng):
            pool1_out, conv1_out, pool2_out, conv2_out, flatten, Z_1, Z_2, Z_3, Z_4 = forward_prop(j, W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4)
            dW_conv1, dB_conv1, dW_conv2, dB_conv2, dW_1, dB_1, dW_2, dB_2, dW_3, dB_3, dW_4, dB_4 = back_prop(pool1_out, conv1_out, pool2_out, conv2_out, flatten, Z_1, Z_2, Z_3, Z_4,
                                                                                                             W_conv1, B_conv1, W_conv2, B_conv2,
                                                                                                             W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4, j)
            W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4 = update_params(
                W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4,
                dW_conv1, dB_conv1, dW_conv2, dB_conv2,
                dW_1, dB_1, dW_2, dB_2, dW_3, dB_3, dW_4, dB_4,
                learning_rate
            )
        # Print cost after each epoch
        print(f"Epoch {i}: cost = {get_cost(Z_4, j)[0]}, LR = {learning_rate}")
    train_end = time.time()
    print(f"Total training time: {(train_end-train_start)} seconds")

    np.savez('model_weights_10_epochs.npz', W_conv1=W_conv1, B_conv1=B_conv1, W_conv2=W_conv2, B_conv2=B_conv2, W_1=W_1, B_1=B_1, W_2=W_2, B_2=B_2, W_3=W_3, B_3=B_3, W_4=W_4, B_4=B_4)

    return W_conv1, B_conv1, W_conv2, B_conv2, W_1, B_1, W_2, B_2, W_3, B_3, W_4, B_4
