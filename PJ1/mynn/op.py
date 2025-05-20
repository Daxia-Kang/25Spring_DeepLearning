from abc import abstractmethod
import numpy as np

def he_initialization_linear(size):
    return np.random.randn(*size) * np.sqrt(2 / size[1])

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
    
class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=he_initialization_linear, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X # Store input for backpropagation 
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        X = self.input
        self.grads['W'] = X.T @ grad
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        grad_input = grad @ self.W.T
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=he_initialization_linear, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size))
        self.b = initialize_method(size=(1, out_channels))
        self.stride = stride
        self.pad = padding
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        #中间数据(backward使用)
        self.X = None
        self.col = None
        self.col_W = None
        #权重和偏置参数的梯度
        self.dW = None
        self.db = None

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        FN, C, FH, FW = self.W.shape #FN输出通道
        N, C, H, W = X.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)
        
        col = im2col(X, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.X = X
        self.col = col
        self.col_W = col_W
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        FN, C, FH, FW = self.W.shape #FN输出通道
        grads = grads.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.grads['b'] = np.sum(grads, axis=0)
        self.grads['W'] = np.dot(self.col.T, grads)
        self.grads['W'] = self.grads['W'].reshape(FN, C, FH, FW)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W

        dcol = np.dot(grads, self.col_W.T)
        grad_input = col2im(dcol, self.X.shape, FH, FW, self.stride, self.pad)

        return grad_input  
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
    
class Pooling(Layer):
    def __init__(self, pool_h, pool_w, stride=2, pad=0) -> None:
        super().__init__()
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.input = None
        self.arg_max = None
        self.optimizable = False

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        N, C, H, W = X.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(X, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis = 1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.input = X
        self.arg_max = arg_max
        
        return out
    
    def backward(self, dout): #将上游梯度 dout（池化层输出的梯度）分配回这些最大值的位置。
        dout = dout.transpose(0, 2, 3, 1) #N, C, out_h, out_w->

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size)) # (N*out_h*out_w*C, pool_size)
        dmax[np.arange(dout.size), self.arg_max] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size, ))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.input.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False #无需优化

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.has_softmax = True
        self.max_classes = max_classes
        self.labels = None
        self.predicts = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels
        batch_size = predicts.shape[0]

        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts

        probs_clipped = np.clip(probs, 1e-8, 1 - 1e-8)
        correct_probs = probs_clipped[np.arange(batch_size), labels]
        loss = -np.sum(np.log(correct_probs)) / batch_size
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        # Then send the grads to model for back propagation
        batch_size = self.predicts.shape[0]
        if self.has_softmax:
            one_hot = np.zeros_like(self.predicts)
            one_hot[np.arange(batch_size), self.labels] = 1
            probs = softmax(self.predicts)
            grad = (probs - one_hot) / batch_size
        else:
            raise NotImplementedError("The backward function of MultiCrossEntropyLoss without softmax is not implemented yet.")
        self.model.backward(grad)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition


