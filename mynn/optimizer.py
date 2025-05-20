from abc import abstractmethod
import numpy as np

class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key]


class MomentumGD(Optimizer):
    def __init__(self, init_lr, model, momentum=0.9):
        super().__init__(init_lr, model)
        self.momentum = momentum
        self.prev_params = {layer: {key: np.zeros_like(param) for key, param in layer.params.items()} for layer in self.model.layers if layer.optimizable}

    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)

                    # Calculate the update
                    param_prev = self.prev_params[layer][key]
                    param_curr = layer.params[key]
                    update = -self.init_lr * layer.grads[key] + self.momentum * (param_curr - param_prev)

                    # Update parameter
                    self.prev_params[layer][key] = param_curr
                    layer.params[key] += update
