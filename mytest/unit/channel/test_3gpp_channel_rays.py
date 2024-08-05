import torch

try:
    import my_code.mysionna
except ImportError as e:
    import sys
    sys.path.append("../")

# Check if GPU is available
if torch.cuda.is_available():
    # Number of GPUs available
    num_gpus = torch.cuda.device_count()
    print('Number of GPUs available:', num_gpus)

    # Specify which GPU to use
    gpu_num = 0 # Number of the GPU to be used

    if gpu_num < num_gpus:
        device = torch.device(f'cuda:{gpu_num}')
        print(f'Using GPU number {gpu_num}.')

        # Set the specified GPU as the current device
        torch.cuda.set_device(device)

        # This functionality is built-in in PyTorch and does not need to be set manually.
        # Memory growth equivalent in PyTorch is handled by default.
    else:
        print(f'GPU number {gpu_num} is not available. Using CPU.')
        device = torch.device('cpu')
else:
    print('No GPU available. Using CPU.')
    device = torch.device('cpu')

import unittest
import numpy as np
import my_code.mysionna as sionna
from mytest.unit.channel.channel_test_utils import *
from scipy.stats import kstest

from  my_code.mysionna.channel.torch_version.tr38901 import antenna


class TestRays(unittest.TestCase):
    r"""Test the rays generated for 3GPP system level simulations
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 3e-2

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 35.0


    def setUpClass():
        r"""Sample rays from all LoS and NLoS channel models for testing"""

        # Forcing the seed to make the tests deterministic
        torch.manual_seed(42)
        np.random.seed(42)

        batch_size = TestRays.BATCH_SIZE
        fc = TestRays.CARRIER_FREQUENCY

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        bs_array = antenna.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)
        ut_array = antenna.PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='single',
                                                    polarization_type='V',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)
        

        # The following quantities have no impact on the rays, but are
        # required to instantiate models
        ut_orientations = torch.rand([batch_size, 1, 3],
                                            dtype=torch.float64)*(2*sionna.PI)+sionna.PI
        bs_orientations = torch.rand([batch_size, 1, 3],
                                            -sionna.PI, sionna.PI,
                                            dtype=torch.float64)*(2*sionna.PI)+sionna.PI
        ut_velocities = torch.rand([batch_size, 1, 3], -1.0, 1.0,
                                            dtype=torch.float64)*2+1

        # 1 UT and 1 BS
        ut_loc = generate_random_loc(batch_size, 1, (100,2000), (100,2000),
                                        (1.5, 1.5), share_loc=True,
                                        dtype=torch.float64)
        bs_loc = generate_random_loc(batch_size, 1, (0,100), (0,100),
                                        (35.0, 35.0), share_loc=True,
                                        dtype=torch.float64)