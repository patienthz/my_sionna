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

from my_code.mysionna.channel.torch_version.tr38901.antenna import PanelArray
from my_code.mysionna.channel.torch_version.tr38901.rma import RMaScenario,RMa


class TestScenario(unittest.TestCase):
    r"""Test the distance calculations and function that get the parameters
    according to the scenario.
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR = 1e-2

    # Heigh of UTs
    H_UT = 1.5

    # Heigh of BSs
    H_BS = 10.0

    # Number of BS
    NB_BS = 3

    # Number of UT
    NB_UT = 10


    def setUp(self):

        # Forcing the seed to make the tests deterministic
        torch.manual_seed(42)

        batch_size = TestScenario.BATCH_SIZE
        nb_bs = TestScenario.NB_BS
        nb_ut = TestScenario.NB_UT
        fc = TestScenario.CARRIER_FREQUENCY
        h_ut = TestScenario.H_UT
        h_bs = TestScenario.H_BS

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        bs_array = PanelArray(num_rows_per_panel=2,
                                    num_cols_per_panel=2,
                                    polarization='dual',
                                    polarization_type='VH',
                                    antenna_pattern='38.901',
                                    carrier_frequency=fc)
        ut_array = PanelArray(num_rows_per_panel=1,
                                    num_cols_per_panel=1,
                                    polarization='dual',
                                    polarization_type='VH',
                                    antenna_pattern='38.901',
                                    carrier_frequency=fc)

        # The following quantities have no impact on LSP
        # However,these are needed to instantiate the model
        ut_orientations = torch.zeros([batch_size, nb_ut])
        bs_orientations = torch.zeros([batch_size, nb_ut])
        ut_velocities = torch.zeros([batch_size, nb_ut])

        self.scenario = RMaScenario(fc, ut_array,
                                        bs_array, "uplink")

        ut_loc = generate_random_loc(batch_size, nb_ut, (100,2000),
                                     (100,2000), (h_ut, h_ut))
        bs_loc = generate_random_loc(batch_size, nb_bs, (0,100),
                                            (0,100), (h_bs, h_bs))

        in_state = generate_random_bool(batch_size, nb_ut, 0.5)
        self.scenario.set_topology(ut_loc, bs_loc, ut_orientations,
                                bs_orientations, ut_velocities, in_state)
        
    def test_dist(self):
        """Test calculation of distances (total, in, and out)"""
        d_3d = self.scenario.distance_3d
        d_3d_in = self.scenario.distance_3d_in
        d_3d_out = self.scenario.distance_3d_out
        d_2d = self.scenario.distance_2d
        d_2d_in = self.scenario.distance_2d_in
        d_2d_out = self.scenario.distance_2d_out
        # Checking total 3D distances
        ut_loc = self.scenario.ut_loc
        bs_loc = self.scenario.bs_loc
        bs_loc = torch.unsqueeze(bs_loc, dim=2)
        ut_loc = torch.unsqueeze(ut_loc, dim=1)
        d_3d_ref = torch.sqrt(torch.sum(torch.square(ut_loc-bs_loc), dim=3))
        max_err = torch.max(torch.abs(d_3d - d_3d_ref)/d_3d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking 3D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_3d-d_3d_in-d_3d_out)/d_3d)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking total 2D distances
        ut_loc = self.scenario.ut_loc
        bs_loc = self.scenario.bs_loc
        bs_loc = torch.unsqueeze(bs_loc, dim=2)
        ut_loc = torch.unsqueeze(ut_loc, dim=1)
        d_2d_ref = torch.sqrt(torch.sum(torch.square(ut_loc[:,:,:,:2]-bs_loc[:,:,:,:2]), dim=3))
        max_err = torch.max(torch.abs(d_2d - d_2d_ref)/d_2d_ref)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking 2D indoor + outdoor = total
        max_err = torch.max(torch.abs(d_2d-d_2d_in-d_2d_out)/d_2d)
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)
        # Checking indoor/outdoor 2d/3d basic proportionality
        ratio_2d = d_2d_in/d_2d
        ratio_3d = d_3d_in/d_3d
        max_err = torch.max(torch.abs(ratio_2d-ratio_3d))
        self.assertLessEqual(max_err, TestScenario.MAX_ERR)

    def test_get_param(self):
        """Test the get_param() function"""
        # Test if muDSc is correctly extracted from the file (RMa)
        param_tensor_ref = np.zeros([TestScenario.BATCH_SIZE,
                                        TestScenario.NB_BS, TestScenario.NB_UT])
        indoor = np.tile(np.expand_dims(self.scenario.indoor.numpy(), axis=1),
                            [1, TestScenario.NB_BS, 1])
        indoor_index = np.where(indoor)
        los_index = np.where(self.scenario.los.numpy())
        nlos_index = np.where(np.logical_not(self.scenario.los.numpy()))
        param_tensor_ref[los_index] = -7.49
        param_tensor_ref[nlos_index] = -7.43
        param_tensor_ref[indoor_index] = -7.47
        #
        param_tensor = self.scenario.get_param('muDSc').numpy()
        max_err = np.max(np.abs(param_tensor-param_tensor_ref))
        self.assertLessEqual(max_err, 1e-6)

if __name__=='__main__':
    unittest.main()