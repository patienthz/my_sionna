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

import my_code.mysionna as sionna
import unittest
import numpy as np
from my_code.mysionna.channel.torch_version.tr38901.tdl import TDL
from my_code.mysionna.channel.torch_version.utils import exp_corr_mat

from scipy.stats import kstest, rayleigh, rice
from scipy.special import jv



class TestTDL(unittest.TestCase):
    r"""Test the 3GPP TDL channel model.
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Frequency at which the channel is sampled
    SAMPLING_FREQUENCY = 15e3 # Hz

    # Delay spread
    DELAY_SPREAD = 100e-9 # s

    #  Number of time steps per example
    NUM_TIME_STEPS = 100

    # Number of sinusoids for channel coefficient generation
    NUM_SINUSOIDS = 20

    # Speed
    SPEED = 150 # m/s
    MAX_DOPPLER = 2.*sionna.PI*SPEED/sionna.SPEED_OF_LIGHT*CARRIER_FREQUENCY

    # AoA
    LoS_AoA = np.pi/4

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 5e-2