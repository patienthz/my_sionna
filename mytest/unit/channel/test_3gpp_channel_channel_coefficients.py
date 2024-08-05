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
from mytest.unit.channel.channel_test_utils import *

from my_code.mysionna.channel.torch_version.tr38901.antenna import PanelArray,Antenna
from my_code.mysionna.channel.torch_version.tr38901.channel_coefficients import ChannelCoefficientsGenerator,Topology
from my_code.mysionna.channel.torch_version.tr38901.rma import RMaScenario
from my_code.mysionna.channel.torch_version.tr38901.lsp import LSPGenerator
from my_code.mysionna.channel.torch_version.tr38901.rays import RaysGenerator

class TestChannelCoefficientsGenerator(unittest.TestCase):
    r"""Test the computation of channel coefficients"""

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 32

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for calculation (relative error)
    MAX_ERR = 1e-2

    # # Heigh of UTs
    H_UT = 1.5

    # # Heigh of BSs
    H_BS = 10.0

    # # Number of BS
    NB_BS = 3

    # Number of UT
    NB_UT = 10

    # Number of channel time samples
    NUM_SAMPLES = 64

    # Sampling frequency
    SAMPLING_FREQUENCY = 20e6

    def setUp(self):

        # Forcing the seed to make the tests deterministic
        torch.manual_seed(42)

        fc = TestChannelCoefficientsGenerator.CARRIER_FREQUENCY

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        self.tx_array = PanelArray(num_rows_per_panel=2,
                                                    num_cols_per_panel=2,
                                                    polarization='dual',
                                                    polarization_type='VH',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)
        self.rx_array = PanelArray(num_rows_per_panel=1,
                                                    num_cols_per_panel=1,
                                                    polarization='dual',
                                                    polarization_type='VH',
                                                    antenna_pattern='38.901',
                                                    carrier_frequency=fc,
                                                    dtype=torch.complex128)

        self.ccg = ChannelCoefficientsGenerator(
            fc,
            tx_array=self.tx_array,
            rx_array=self.rx_array,
            subclustering=True,
            dtype=torch.complex128)

        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        nb_ut = TestChannelCoefficientsGenerator.NB_UT
        nb_bs = TestChannelCoefficientsGenerator.NB_BS
        h_ut = TestChannelCoefficientsGenerator.H_UT
        h_bs = TestChannelCoefficientsGenerator.H_BS

        rx_orientations = torch.rand((batch_size, nb_ut, 3), dtype=torch.float64) * 2 * np.pi
        tx_orientations = torch.rand((batch_size, nb_bs, 3), dtype=torch.float64) * 2 * np.pi
        ut_velocities = torch.rand((batch_size, nb_ut, 3), dtype=torch.float64) * 5.0


        scenario = RMaScenario(fc, self.rx_array,
                                    self.tx_array,
                                    "downlink",
                                    dtype=torch.complex128)

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                     (100,2000), (h_ut, h_ut), dtype=torch.float64)
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs),
                                            dtype=torch.float64)

        in_state = generate_random_bool(batch_size, nb_ut, 0.5)
        scenario.set_topology(ut_loc, bs_loc, rx_orientations,
                                tx_orientations, ut_velocities, in_state)
        self.scenario = scenario

        topology = Topology(
            velocities=ut_velocities,
            moving_end='rx',
            los_aoa=scenario.los_aoa,
            los_aod=scenario.los_aod,
            los_zoa=scenario.los_zoa,
            los_zod=scenario.los_zod,
            los=scenario.los,
            distance_3d=scenario.distance_3d,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations)
        self.topology = topology

        lsp_sampler = LSPGenerator(scenario)
        ray_sampler = RaysGenerator(scenario)
        lsp_sampler.topology_updated_callback()
        ray_sampler.topology_updated_callback()
        lsp = lsp_sampler()
        self.rays = ray_sampler(lsp)
        self.lsp = lsp

        num_time_samples = TestChannelCoefficientsGenerator.NUM_SAMPLES
        sampling_frequency = TestChannelCoefficientsGenerator.SAMPLING_FREQUENCY
        c_ds = scenario.get_param("cDS")*1e-9
        _, _, phi, sample_times = self.ccg(num_time_samples,
            sampling_frequency, lsp.k_factor, self.rays, topology, c_ds,
            debug=True)
        self.phi = phi.numpy()
        self.sample_times = sample_times.numpy()
        self.c_ds = c_ds

    def max_rel_err(self, r, x):
        """Compute the maximum relative error, ``r`` being the reference value,
        ``x`` an esimate of ``r``."""
        err = np.abs(r-x)
        rel_err = np.where(np.abs(r) > 0.0, np.divide(err,np.abs(r)+1e-6), err)
        return np.max(rel_err)

    def unit_sphere_vector_ref(self, theta, phi):
        """Reference implementation: Unit to sphere vector"""
        uvec = np.stack([np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi), np.cos(theta)],
                            axis=-1)
        uvec = np.expand_dims(uvec, axis=-1)
        return uvec

    def test_unit_sphere_vector(self):
        """Test 3GPP channel coefficient calculation: Unit sphere vector"""
        #
        batch_size = TestChannelCoefficientsGenerator.BATCH_SIZE
        theta = torch.randn(batch_size).numpy()
        phi = torch.randn(batch_size).numpy()
        uvec_ref = self.unit_sphere_vector_ref(theta, phi)
        uvec = self.ccg._unit_sphere_vector(theta, phi).numpy()
        max_err = self.max_rel_err(uvec_ref, uvec)
        err_tol = TestChannelCoefficientsGenerator.MAX_ERR
        self.assertLessEqual(max_err, err_tol)