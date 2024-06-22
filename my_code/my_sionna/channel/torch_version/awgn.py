import torch
import torch.nn as nn
import torch.nn.functional as F

class AWGN(nn.Module):
    r"""AWGN(dtype=torch.complex64, **kwargs)

    Add complex AWGN to the inputs with a certain variance.

    This class inherits from the PyTorch `nn.Module` class and can be used as a layer in
    a PyTorch model.

    This layer adds complex AWGN noise with variance `no` to the input.
    The noise has variance `no/2` per real dimension.
    It can be either a scalar or a tensor which can be broadcast to the shape
    of the input.

    Parameters
    ----------
        dtype : Complex torch.dtype
            Defines the datatype for internal calculations and the output
            dtype. Defaults to `torch.complex64`.

    Input
    -----

        (x, no) :
            Tuple:

        x :  Tensor, torch.complex
            Channel input

        no : Scalar or Tensor, torch.float
            Scalar or tensor whose shape can be broadcast to the shape of `x`.
            The noise power `no` is per complex dimension. If `no` is a
            scalar, noise of the same variance will be added to the input.
            If `no` is a tensor, it must have a shape that can be broadcast to
            the shape of `x`. This allows, e.g., adding noise of different
            variance to each example in a batch. If `no` has a lower rank than
            `x`, then `no` will be broadcast to the shape of `x` by adding
            dummy dimensions after the last axis.

    Output
    -------
        y : Tensor with same shape as `x`, torch.complex
            Channel output
    """

    def __init__(self, dtype=torch.complex64, **kwargs):
        super().__init__()
        self.dtype = dtype

    def forward(self, inputs):
        x, no = inputs

        # Create real-valued Gaussian noise for each complex dimension
        noise_real = torch.randn_like(x.real)
        noise_imag = torch.randn_like(x.imag)
        noise = noise_real + 1j * noise_imag

        # Scale noise according to variance
        scale = torch.sqrt(no / 2)
        noise *= scale.unsqueeze(-1)  # broadcast over last dimension

        # Convert noise to complex tensor
        noise = torch.complex(noise.real, noise.imag)

        # Add noise to input
        y = x + noise.to(self.dtype)

        return y