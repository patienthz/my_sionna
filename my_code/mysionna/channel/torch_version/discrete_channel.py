import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryMemorylessChannel(nn.Module):
    def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100., dtype=torch.float32, **kwargs):
        super().__init__(**kwargs)

        assert isinstance(return_llrs, bool), "return_llrs must be bool."
        self.return_llrs = return_llrs

        assert isinstance(bipolar_input, bool), "bipolar_input must be bool."
        self.bipolar_input = bipolar_input

        assert llr_max >= 0., "llr_max must be a positive scalar value."
        self.llr_max = llr_max
        self.dtype = dtype

        if self.return_llrs:
            assert dtype in (torch.float16, torch.float32, torch.float64), \
                "LLR outputs require non-integer dtypes."
        else:
            if self.bipolar_input:
                assert dtype in (torch.float16, torch.float32, torch.float64,
                                 torch.int8, torch.int16, torch.int32, torch.int64), \
                    "Only signed dtypes are supported for bipolar inputs."
            else:
                assert dtype in (torch.float16, torch.float32, torch.float64,
                                 torch.uint8, torch.uint16, torch.uint32, torch.uint64,
                                 torch.int8, torch.int16, torch.int32, torch.int64), \
                    "Only real-valued dtypes are supported."

        self.check_input = True  # check input for consistency (i.e., binary)

        self.eps = 1e-9  # small additional term for numerical stability
        self.temperature = torch.tensor(0.1, dtype=torch.float32)  # for Gumble-softmax

    @property
    def llr_max(self):
        """Maximum value used for LLR calculations."""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value):
        """Maximum value used for LLR calculations."""
        assert value >= 0, 'llr_max cannot be negative.'
        self._llr_max = value

    @property
    def temperature(self):
        """Temperature for Gumble-softmax trick."""
        return self._temperature.item()

    @temperature.setter
    def temperature(self, value):
        """Temperature for Gumble-softmax trick."""
        assert value >= 0, 'temperature cannot be negative.'
        self._temperature = torch.tensor(value, dtype=torch.float32)

    def _check_inputs(self, x):
        """Check input x for consistency, i.e., verify
        that all values are binary of bipolar values."""
        x = x.float()
        if self.check_input:
            if self.bipolar_input:
                assert torch.all(torch.logical_or(x == -1, x == 1)), "Input must be bipolar {-1, 1}."
            else:
                assert torch.all(torch.logical_or(x == 0, x == 1)), "Input must be binary {0, 1}."
            # input datatype consistency should be only evaluated once
            self.check_input = False

    def _custom_xor(self, a, b):
        """Straight through estimator for XOR."""
        return torch.abs(a - b)

    def _ste_binarizer(self, x):
        """Straight through binarizer to quantize bits to int values."""
        return torch.where(x < 0.5, torch.tensor(0., dtype=self.dtype), torch.tensor(1., dtype=self.dtype))

    def _sample_errors(self, pb, shape):
        """Samples binary error vector with given error probability e.
        This function is based on the Gumble-softmax "trick" to keep the
        sampling differentiable."""

        u1 = torch.rand(shape)
        u2 = torch.rand(shape)
        u = torch.stack((u1, u2), dim=-1)

        # sample Gumble distribution
        q = -torch.log(-torch.log(u + self.eps) + self.eps)
        p = torch.stack((pb, 1 - pb), dim=-1)
        p = p.unsqueeze(0).expand(q.shape)
        a = (torch.log(p + self.eps) + q) / self.temperature

        # apply softmax
        e_cat = F.softmax(a, dim=-1)

        # binarize final values via straight-through estimator
        return self._ste_binarizer(e_cat[..., 0])  # only take the first class

    def forward(self, x, pb):
        # allow pb to be a tuple of two scalars
        if isinstance(pb, (tuple, list)):
            pb0 = pb[0]
            pb1 = pb[1]
        else:
            pb0 = pb[..., 0]
            pb1 = pb[..., 1]

        # clip for numerical stability
        pb0 = torch.clamp(pb0, 0., 1.).float()
        pb1 = torch.clamp(pb1, 0., 1.).float()

        # check x for consistency (binary, bipolar)
        self._check_inputs(x)

        e0 = self._sample_errors(pb0, x.shape)
        e1 = self._sample_errors(pb1, x.shape)

        if self.bipolar_input:
            neutral_element = torch.tensor(-1., dtype=self.dtype)
        else:
            neutral_element = torch.tensor(0., dtype=self.dtype)

        e = torch.where(x == neutral_element, e0, e1).type(self.dtype)

        if self.bipolar_input:
            y = x * (-2 * e + 1)
        else:
            y = self._custom_xor(x, e)

        if self.return_llrs:
            if not self.bipolar_input:
                y = 2 * y - 1  # transform to bipolar

            y0 = - (torch.log(pb1 + self.eps) - torch.log(1 - pb0 - self.eps))
            y1 = (torch.log(1 - pb1 - self.eps) - torch.log(pb0 + self.eps))
            y = torch.where(y == 1, y1, y0) * y
            y = torch.clamp(y, -self.llr_max, self.llr_max)

        return y
