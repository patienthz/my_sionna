import torch
import torch.nn as nn
import numpy as np
from my_code.mysionna.mapping import  Mapper, Constellation

def complex_normal(shape, var=1.0, dtype=torch.complex64):
    # Half the variance for each dimension
    if dtype == torch.complex32:
        real_dtype =  torch.float16
    if dtype == torch.complex64:
        real_dtype = torch.float32
    if dtype == torch.complex128:
        real_dtype = torch.float64

    
    var_dim = torch.tensor(var,dtype=real_dtype)/torch.tensor(2,dtype=real_dtype)
    stddev = torch.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = torch.normal(mean=0,std=stddev, size=shape, dtype= real_dtype)
    xi = torch.normal(mean=0,std=stddev, size=shape, dtype= real_dtype)
    x = torch.view_as_complex(torch.stack((xr, xi), axis=-1))

    return x


class BinarySource(nn.Module):
    """
    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : torch.dtype
        Defines the output datatype of the layer.
        Defaults to `torch.float32`.

    seed : int or None
        Set the seed for the random generator used to generate the bits.
        Set to `None` for random initialization of the RNG.

    device : torch.device or None
        The device on which the tensor will be created.
        Defaults to `None`, which means the tensor will be created on the CPU.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor filled with random binary values.
    """
    def __init__(self, dtype=torch.float32, seed=None, device=None, **kwargs):
        super().__init__(**kwargs)
        self.dtype = dtype
        self._seed = seed
        self.device = device if device is not None else torch.device('cpu')
        self._rng = torch.Generator(device=self.device)
        if self._seed is not None:
            self._rng.manual_seed(self._seed)

    def forward(self, inputs):
        if self._seed is not None:
            return torch.randint(0, 2, tuple(inputs.tolist()), generator=self._rng, dtype=torch.int32, device=self.device).to(self.dtype)
        else:
            return torch.randint(0, 2, tuple(inputs.tolist()), dtype=torch.int32, device=self.device).to(self.dtype)

class SymbolSource(nn.Module):
    """
    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [torch.complex64, torch.complex128], torch.dtype
        The output dtype. Defaults to torch.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, torch.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], torch.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 device= "cuda",
                 **kwargs):
        super().__init__(**kwargs)
        constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype=dtype)
        
        if dtype ==torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("real_dtype must be in [torch.float16,torch.float32,torch.float64]")
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, dtype=real_dtype,device=device)
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype,
                              device=device)

    def forward(self, inputs):
        if not isinstance(inputs,torch.Tensor):
            inputs = torch.tensor(inputs)
        shape = torch.cat([inputs, torch.tensor([self._num_bits_per_symbol])], dim=-1)
        b = self._binary_source(shape.to(torch.int32))
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        result = x.squeeze(-1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(ind.squeeze(-1))
        if self._return_bits:
            result.append(b)

        return result


class QAMSource(SymbolSource):
    # pylint: disable=line-too-long
    r"""QAMSource(num_bits_per_symbol=None, return_indices=False, return_bits=False, seed=None, dtype=torch.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random QAM symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [torch.complex64, torch.complex128], torch.DType
        The output dtype. Defaults to torch.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random QAM symbols.

    symbol_indices : ``shape``, torch.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], torch.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=torch.complex64,
                 device=None,
                 **kwargs):
        super(QAMSource, self).__init__(constellation_type="qam",
                                        num_bits_per_symbol=num_bits_per_symbol,
                                        return_indices=return_indices,
                                        return_bits=return_bits,
                                        seed=seed,
                                        dtype=dtype,
                                        device=device,
                                        **kwargs)


