import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import my_code.mysionna as sn
def gather_pytorch(input_data,  axis=0, indices=None,device=None):
    if axis < 0:
        axis = len(input_data.shape) + axis
    indices = indices.to(input_data.device)    
    data = torch.index_select(input_data, axis, indices.flatten())

    shape_input = list(input_data.shape)
    # shape_ = delete(shape_input, axis)
    
    # 连接列表
    shape_output = shape_input[:axis] + \
        list(indices.shape) + shape_input[axis + 1:]

    data_output = data.reshape(shape_output)

    return data_output

def pam_gray(b):
    # pylint: disable=line-too-long
    r"""Maps a vector of bits to a PAM constellation points with Gray labeling.

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generated QAM constellations.
    The constellation is not normalized.

    Input
    -----
    b : [n], NumPy array
        Tensor with with binary entries.

    Output
    ------
    : signed int
        The PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    Note
    ----
    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    if len(b)>1:
        return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
    return 1-2*b[0]

def qam(num_bits_per_symbol, normalize=True):
    r"""Generates a QAM constellation.

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.complex64
        The QAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol % 2 == 0 # is even
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be a multiple of 2") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.complex64)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dimension

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(qam_var)
    return c

def pam(num_bits_per_symbol, normalize=True):
    r"""Generates a PAM constellation.

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : int
        The number of bits per constellation point.
        Must be positive.

    normalize: bool
        If `True`, the constellation is normalized to have unit power.
        Defaults to `True`.

    Output
    ------
    : :math:`[2^{\text{num_bits_per_symbol}}]`, np.float32
        The PAM constellation.

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be positive") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=np.float32)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int16)
        c[i] = pam_gray(b)

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = 1/(2**(n-1))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
        c /= np.sqrt(pam_var)
    return c

class Constellation(nn.Module):
    def __init__(self,
                 constellation_type,
                 num_bits_per_symbol,
                 initial_value=None,
                 normalize=True,
                 center=False,
                 trainable=False,
                 dtype=torch.complex64,
                 device='cuda',
                 **kwargs):
        super(Constellation, self).__init__()
        assert dtype in [torch.complex64, torch.complex128],\
            "dtype must be torch.complex64 or torch.complex128"
        self._dtype = dtype
        self.device = device

        # 获得 real_dtype
        if dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("dtype must be torch.complex64 or torch.complex128")
        # 检查参数异常
        assert constellation_type in ("qam", "pam", "custom"),\
            "Wrong constellation type"
        self._constellation_type = constellation_type

        assert isinstance(normalize, bool), "normalize must be boolean"
        self._normalize = normalize

        assert isinstance(center, bool), "center must be boolean"
        self._center = center

        assert isinstance(trainable, bool), "trainable must be boolean"
        self._trainable = trainable

        assert isinstance(num_bits_per_symbol, (float, int)),\
            "num_bits_per_symbol must be integer"
        assert (num_bits_per_symbol % 1 == 0),\
            "num_bits_per_symbol must be integer"
        num_bits_per_symbol = int(num_bits_per_symbol)

        #不同的 constellation_type
        if self._constellation_type == "qam":
            assert num_bits_per_symbol % 2 == 0 and num_bits_per_symbol > 0,\
                "num_bits_per_symbol must be a multiple of 2"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            assert initial_value is None, "QAM must not have an initial value"
            points = qam(self._num_bits_per_symbol, normalize=self._normalize)
            if not isinstance(points,torch.Tensor):
                points = torch.tensor(points)
            points = points.to(self._dtype).to(self.device)

        elif self._constellation_type == "pam":
            assert num_bits_per_symbol > 0,\
                "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = num_bits_per_symbol

            assert initial_value is None, "PAM must not have an initial value"
            points = pam(self._num_bits_per_symbol, normalize=self._normalize)
            if not isinstance(points,torch.Tensor):
                points = torch.tensor(points)
            points = points.to(self._dtype).to(self.device)

        elif self._constellation_type == "custom":
            assert num_bits_per_symbol > 0,\
                "num_bits_per_symbol must be integer"
            self._num_bits_per_symbol = int(num_bits_per_symbol)

            if initial_value is None:
                points = torch.rand(2, 2 ** self._num_bits_per_symbol, dtype=real_dtype, device=self.device) * 0.1 - 0.05
                points = torch.complex(points[0], points[1])
            else:
                if not isinstance(initial_value,torch.Tensor):
                    initial_value = torch.tensor(initial_value)
                assert initial_value.dim().numpy()== 1
                assert initial_value.shape[0] == 2 ** num_bits_per_symbol,\
                    "initial_value must have shape [2**num_bits_per_symbol]"
                points = initial_value.to(self._dtype).to(self.device)
        self._points = points

        if self._trainable:
            self._points = nn.Parameter(self._points)
    # pylint: disable=no-self-argument
    def create_or_check_constellation(  constellation_type=None,
                                        num_bits_per_symbol=None,
                                        constellation=None,
                                        dtype=torch.complex64):
        # pylint: disable=line-too-long
        r"""Static method for conviently creating a constellation object or checking that an existing one
        is consistent with requested settings.

        If ``constellation`` is `None`, then this method creates a :class:`~sionna.mapping.Constellation`
        object of type ``constellation_type`` and with ``num_bits_per_symbol`` bits per symbol.
        Otherwise, this method checks that `constellation` is consistent with ``constellation_type`` and
        ``num_bits_per_symbol``. If it is, ``constellation`` is returned. Otherwise, an assertion is raised.

        Input
        ------
        constellation_type : One of ["qam", "pam", "custom"], str
            For "custom", an instance of :class:`~sionna.mapping.Constellation`
            must be provided.

        num_bits_per_symbol : int
            The number of bits per constellation symbol, e.g., 4 for QAM16.
            Only required for ``constellation_type`` in ["qam", "pam"].

        constellation :  Constellation
            An instance of :class:`~sionna.mapping.Constellation` or
            `None`. In the latter case, ``constellation_type``
            and ``num_bits_per_symbol`` must be provided.

        Output
        -------
        : :class:`~sionna.mapping.Constellation`
            A constellation object.
        """
        constellation_object = None
        if constellation is not None:
            assert constellation_type in [None, "custom"], \
                """`constellation_type` must be "custom"."""
            assert num_bits_per_symbol in \
                     [None, constellation.num_bits_per_symbol], \
                """`Wrong value of `num_bits_per_symbol.`"""
            assert constellation.dtype==dtype, \
                "Constellation has wrong dtype."
            constellation_object = constellation
        else:
            assert constellation_type in ["qam", "pam"], \
                "Wrong constellation type."
            assert num_bits_per_symbol is not None, \
                "`num_bits_per_symbol` must be provided."
            constellation_object = Constellation(   constellation_type,
                                                    num_bits_per_symbol,
                                                    dtype=dtype)
        return constellation_object
    def forward(self, x):
        x = self._points
        if self._center:
            x = x - x.mean()
        if self._normalize:
            energy = (x.abs() ** 2).mean()
            energy_sqrt = torch.sqrt(energy).to(self._dtype)
            x = x / energy_sqrt
        return x

    @property
    def normalize(self):
        """Indicates if the constellation is normalized or not."""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        assert isinstance(value, bool), "`normalize` must be boolean"
        self._normalize = value

    @property
    def center(self):
        """Indicates if the constellation is centered."""
        return self._center

    @center.setter
    def center(self, value):
        assert isinstance(value, bool), "`center` must be boolean"
        self._center = value

    @property
    def num_bits_per_symbol(self):
        """The number of bits per constellation symbol."""
        return self._num_bits_per_symbol
    
    @property
    def dtype(self):
        return self._dtype

    @property
    def points(self):
        """The (possibly) centered and normalized constellation points."""
        return self(None)

    def show(self, labels=True, figsize=(7,7)):
        """Generate a scatter-plot of the constellation.

        Input
        -----
        labels : bool
            If `True`, the bit labels will be drawn next to each constellation
            point. Defaults to `True`.

        figsize : Two-element Tuple, float
            Width and height in inches. Defaults to `(7,7)`.

        Output
        ------
        : matplotlib.figure.Figure
            A handle to a matplot figure object.
        """
        maxval = self._points.abs().max().item() * 1.05
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(-maxval, maxval)
        ax.set_ylim(-maxval, maxval)
        points_cpu = self._points.detach().cpu().numpy()
        ax.scatter(points_cpu.real, points_cpu.imag)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Real Part")
        ax.set_ylabel("Imaginary Part")
        ax.grid(True, which="both", axis="both")
        ax.set_title("Constellation Plot")
        if labels:
            for j, p in enumerate(points_cpu):
                ax.annotate(
                    format(j, f'0{self._num_bits_per_symbol}b'),
                    (p.real, p.imag)
                )
        plt.show()
        
class Mapper(nn.Module):
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 dtype=torch.complex64,
                 device='cuda'):
        super(Mapper, self).__init__()

        assert dtype in [torch.complex64, torch.complex128], "dtype must be torch.complex64 or torch.complex128"

        # Create constellation object
        self._constellation = Constellation.create_or_check_constellation(
                                                        constellation_type,
                                                        num_bits_per_symbol,
                                                        constellation,
                                                        dtype=dtype)

        self._return_indices = return_indices
        self.device = device

        self._binary_base = 2**torch.arange(self._constellation.num_bits_per_symbol-1, -1, -1, device=device, dtype=torch.int32)

    @property
    def constellation(self):
        """The Constellation used by the Mapper."""
        return self._constellation

    def forward(self, inputs):
        if not isinstance(inputs,torch.Tensor):
            inputs = torch.tensor(inputs)
        assert inputs.dim() >= 2, "The input must have at least rank 2"

        # Reshape inputs to the desired format
        new_shape = list(inputs.shape[:-1]) + [int(inputs.shape[-1] / self.constellation.num_bits_per_symbol), self.constellation.num_bits_per_symbol]
        inputs_reshaped = inputs.view(new_shape).to(torch.int32)

        # Convert the last dimension to an integer
        int_rep = torch.sum(inputs_reshaped * self._binary_base, dim=-1)
        # Map integers to constellation symbols
        x = gather_pytorch(self.constellation.points,axis=0,indices=int_rep)

        if self._return_indices:
            return x, int_rep
        else:
            return x
