import numpy as np
from regex import W
from functions import Add, Mul, Pow


class Tensor:
    def __init__ (self, data, requires_grad=False, _children=(), _op='', label=''):
        self.data =  data
        self.requires_grad = requires_grad
        self.grad = 0
        self.grad_fn = None
        self._prev = set(_children)
        self._op = _op
        self.label = label

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

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Add.apply(self, -other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self * other


    def __truediv__(self, other):
        return self * other ** -1
    
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise ValueError(f'Tensor power can only be int or float')

        return Pow.apply(self, other)