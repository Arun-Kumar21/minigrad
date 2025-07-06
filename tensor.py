import numpy as np

from functions import Add

class Tensor:
    def __init__ (self, data, requires_grad=False):
        self.data =  data
        self.requires_grad = requires_grad
        self.grad = 0
        self.grad_fn = None

    def __repr__(self):
        return f"tensor(data={self.data}, requires_grad={self.requires_grad})"

    def backward(self, grad_output=None):
        if grad_output is None:
            grad_output = 1.0

        self.grad = grad_output

        if self.grad_fn:
            self.grad_fn.backward(grad_output)
        
    @property
    def dtype(self):
        if isinstance(self.data, np.ndarray):
            return self.data.dtype

        else:
            return np.array(self.data).dtype

    @property
    def shape(self):
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        else:
            return ()
        
    
    def item(self):
        if isinstance(self.data, np.ndarray):
            if self.data.size != 1:
                raise ValueError(f"a Tensor with {self.data.size} elements cannnot converted into scalar")
            return self.data.item()
        else:
            return self.data


    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Add.apply(self, other)
