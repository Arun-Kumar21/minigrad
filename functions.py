from tensor import Tensor

class Function:
    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

class Add(Function):
    @staticmethod
    def apply(a, b):
        output = Tensor(a.data + b.data, _children=(a, b), _op='+')
        output.requires_grad = a.requires_grad or b.requires_grad
        output.grad_fn = AddCtx(a, b)
        return output

class AddCtx:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad_output):
        if self.a.requires_grad:
            self.a.backward(grad_output * 1.0)
        if self.b.requires_grad:
            self.b.backward(grad_output * 1.0)

class Mul(Function):
    @staticmethod
    def apply(a, b):
        output = Tensor(a.data * b.data, _children=(a, b), _op='*')
        output.requires_grad = a.requires_grad or b.requires_grad
        output.grad_fn = MulCtx(a, b)
        return output


class MulCtx:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad_output):
        if self.a.requires_grad:
            self.a.backward(grad_output * self.b.data)
        if self.b.requires_grad:
            self.b.backward(grad_output * self.a.data)

class Pow(Function):
    @staticmethod
    def apply(a, b):
        output = Tensor(a.data ** b, _children=(a,), _op="**")
        output.requires_grad = a.requires_grad

        output.grad_fn = PowCtx(a, b)
        return output

class PowCtx:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad_output):
        if self.a.requires_grad:
            self.a.backward((self.b * self.a.data ** (self.b - 1)) * grad_output)