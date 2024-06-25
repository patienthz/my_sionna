import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg
from my_code.mysionna.channel.torch_version.awgn import AWGN

class ApplyTimeChannel(nn.Module):
    def __init__(self, num_time_samples, l_tot, add_awgn=True, dtype=torch.complex64):
        super(ApplyTimeChannel, self).__init__()
        self.add_awgn = add_awgn
        self.dtype = dtype

        first_column = np.concatenate([np.arange(0, num_time_samples), np.full([l_tot - 1], num_time_samples)])
        first_row = np.concatenate([[0], np.full([l_tot - 1], num_time_samples)])
        self.g = torch.tensor(scipy.linalg.toeplitz(first_column, first_row), dtype=torch.long)

        if self.add_awgn:
            self.awgn = AWGN(dtype=dtype)

    def forward(self, inputs):
        if self.add_awgn:
            x, h_time, no = inputs
        else:
            x, h_time = inputs

        # Prepare the channel input for broadcasting and matrix multiplication
        x = F.pad(x, (0, 1))
        x = x.unsqueeze(2)

        # Gather operation similar to TensorFlow's gather
        x = torch.gather(x, -1, self.g.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(x.size(0), x.size(1), x.size(2), -1, -1))

        # Apply the channel response
        y = torch.sum(h_time * x, dim=-1)
        y = torch.sum(torch.sum(y, dim=4), dim=3)

        # Add AWGN if requested
        if self.add_awgn:
            y = self.awgn((y, no))

        return y

