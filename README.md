# Minigrad [Developing]

A minimal implementation of automatic differentiation (autograd) inspired by PyTorch's autograd system.

## Implementation Details
The autograd system works by:

- Building a computation graph during forward pass.
- Tracking operations and their gradients.
- Computing gradients via backpropagation using the chain rule.

Each tensor maintains references to its parent operations, enabling automatic gradient computation when `.backward()` is called.


### Blog post - **SOON**
I'll write a detailed blog post about this project covering the implementation details.