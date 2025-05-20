from .op import *
from collections import OrderedDict
import pickle
    
class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list # 告知维度信息
        self.act_func = act_func
        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplemented
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        
class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list # 告知维度信息
        self.act_func = act_func
        self.layers = []
        for i in range(len(self.size_list) - 1):
            if i == 0 or i == 1:
                layer = conv2D(in_channels=self.size_list[i], out_channels=self.size_list[i + 1], kernel_size=3)
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
            elif i == 2:
                layer = Pooling(pool_h=2, pool_w=2)
            else:
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
            self.layers.append(layer)
            if i < len(self.size_list) - 2 and i != 2:
                self.layers.append(ReLU())

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        
        for i, layer in enumerate(self.layers):
            # Pooling 层之后展平
            if isinstance(layer, Linear) and i > 0 and isinstance(self.layers[i - 1], Pooling):
                # 记录展平前形状
                self.flatten_shape = outputs.shape[1:]  # (C, H, W)
                outputs = outputs.reshape(outputs.shape[0], -1)
            
            outputs = layer(outputs)
        
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if isinstance(self.layers[i], Pooling):
                # 恢复形状：在 forward 中保存的展平前形状
                C, H, W = self.flatten_shape
                grads = grads.reshape(grads.shape[0], C, H, W)
            # 继续反向传播
            grads = layer.backward(grads)
            '''
            # 新增：打印梯度信息
            if hasattr(layer, 'grads') and layer.optimizable:
                print(f"== Layer: {layer.__class__.__name__} ==")
                for key in layer.grads:
                    grad = layer.grads[key]
                    if grad is not None:
                        print(f"  {key}梯度均值: {np.mean(grad):.6e}, 标准差: {np.std(grad):.6e}")
                        print(f"  {key}梯度范围: [{np.min(grad):.6e}, {np.max(grad):.6e}]")
                    else:
                        print(f"  {key}梯度未计算")
            '''
        return grads
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        
        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        for i in range(len(self.size_list) - 1):
            if i == 0 or i == 1:
                layer = conv2D(in_channels=self.size_list[i], out_channels=self.size_list[i + 1], kernel_size=3)
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i + 2]['lambda']
            elif i == 2:
                layer = Pooling(pool_h=2, pool_w=2)
            else:
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 1]['W']
                layer.b = param_list[i + 1]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 1]['weight_decay']
                layer.weight_decay_lambda = param_list[i + 1]['lambda'] # i + 1 池化层无参数

            self.layers.append(layer)
            if i < len(self.size_list) - 2 and i != 2:
                self.layers.append(ReLU())
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
   


    
    