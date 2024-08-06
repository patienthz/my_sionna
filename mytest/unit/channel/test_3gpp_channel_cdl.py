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
from my_code.mysionna.channel.torch_version.tr38901.cdl import CDL 
from my_code.mysionna.channel.torch_version.tr38901.antenna import PanelArray

from mytest.unit.channel.channel_test_utils import *

class TestCDL(unittest.TestCase):
    r"""Test the 3GPP CDL channel model
    """

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10000

    # Carrier frequency
    CARRIER_FREQUENCY = 3.5e9 # Hz

    # Frequency at which the channel is sampled
    SAMPLING_FREQUENCY = 15e3 # Hz

    # Delay spread
    DELAY_SPREAD = 100e-9 # s

    # Maximum allowed deviation for distance calculation (absolute error)
    MAX_ERR = 1e-4

    # Maximum allowed deviation for distance calculation (relative error)
    MAX_ERR_REL = 1e-2

    def setUpClass():

        # Forcing the seed to make the tests deterministic
        torch.manual_seed(42)
        np.random.seed(42)

        # Dict for storing the samples
        TestCDL.powers = {}
        TestCDL.delays = {}
        TestCDL.aod = {}
        TestCDL.aoa = {}
        TestCDL.zod = {}
        TestCDL.zoa = {}
        TestCDL.xpr = {}

        # UT and BS arrays have no impact on LSP
        # However, these are needed to instantiate the model
        tx_array = PanelArray(  num_rows_per_panel=1,
                                num_cols_per_panel=1,
                                polarization='single',
                                polarization_type='V',
                                antenna_pattern='omni',
                                carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                                dtype=torch.complex128)
        rx_array = PanelArray(  num_rows_per_panel=1,
                                num_cols_per_panel=1,
                                polarization='single',
                                polarization_type='V',
                                antenna_pattern='omni',
                                carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                                dtype=torch.complex128)

        ########## CDL-A
        cdl = CDL(  "A",
                    delay_spread=TestCDL.DELAY_SPREAD,
                    carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                    ut_array=rx_array,
                    bs_array=tx_array,
                    direction='downlink',
                    dtype=torch.complex128)
        a,tau = cdl(100000, 1, 100e6)
        a = a[:,0,0,0,0,:,0].numpy()
        tau = tau.numpy()[0,0,0]
        p = np.mean(np.square(np.abs(a)), axis=0)
        TestCDL.powers['A'] = p
        TestCDL.delays['A'] = tau
        TestCDL.aod['A'] = cdl._aod.numpy()[0,0,0]
        TestCDL.aoa['A'] = cdl._aoa.numpy()[0,0,0]
        TestCDL.zod['A'] = cdl._zod.numpy()[0,0,0]
        TestCDL.zoa['A'] = cdl._zoa.numpy()[0,0,0]
        TestCDL.xpr['A'] = cdl._xpr.numpy()[0,0,0]

        ########## CDL-B
        cdl = CDL(  "B",
                    delay_spread=TestCDL.DELAY_SPREAD,
                    carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                    ut_array=rx_array,
                    bs_array=tx_array,
                    direction='downlink',
                    dtype=torch.complex128)
        a,tau = cdl(100000, 1, 100e6)
        a = a[:,0,0,0,0,:,0].numpy()
        tau = tau.numpy()[0,0,0]
        p = np.mean(np.square(np.abs(a)), axis=0)
        TestCDL.powers['B'] = p
        TestCDL.delays['B'] = tau
        TestCDL.aod['B'] = cdl._aod.numpy()[0,0,0]
        TestCDL.aoa['B'] = cdl._aoa.numpy()[0,0,0]
        TestCDL.zod['B'] = cdl._zod.numpy()[0,0,0]
        TestCDL.zoa['B'] = cdl._zoa.numpy()[0,0,0]
        TestCDL.xpr['B'] = cdl._xpr.numpy()[0,0,0]

        ########## CDL-C
        cdl = CDL(  "C",
                    delay_spread=TestCDL.DELAY_SPREAD,
                    carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                    ut_array=rx_array,
                    bs_array=tx_array,
                    direction='downlink',
                    dtype=torch.complex128)
        a,tau = cdl(100000, 1, 100e6)
        a = a[:,0,0,0,0,:,0].numpy()
        tau = tau.numpy()[0,0,0]
        p = np.mean(np.square(np.abs(a)), axis=0)
        TestCDL.powers['C'] = p
        TestCDL.delays['C'] = tau
        TestCDL.aod['C'] = cdl._aod.numpy()[0,0,0]
        TestCDL.aoa['C'] = cdl._aoa.numpy()[0,0,0]
        TestCDL.zod['C'] = cdl._zod.numpy()[0,0,0]
        TestCDL.zoa['C'] = cdl._zoa.numpy()[0,0,0]
        TestCDL.xpr['C'] = cdl._xpr.numpy()[0,0,0]

        ########## CDL-D
        cdl = CDL(  "D",
                    delay_spread=TestCDL.DELAY_SPREAD,
                    carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                    ut_array=rx_array,
                    bs_array=tx_array,
                    direction='downlink',
                    dtype=torch.complex128)
        a,tau = cdl(100000, 1, 100e6)
        a = a[:,0,0,0,0,:,0].numpy()
        tau = tau.numpy()[0,0,0]
        p = np.mean(np.square(np.abs(a)), axis=0)
        TestCDL.powers['D'] = p
        TestCDL.delays['D'] = tau
        TestCDL.aod['D'] = cdl._aod.numpy()[0,0,0]
        TestCDL.aoa['D'] = cdl._aoa.numpy()[0,0,0]
        TestCDL.zod['D'] = cdl._zod.numpy()[0,0,0]
        TestCDL.zoa['D'] = cdl._zoa.numpy()[0,0,0]
        TestCDL.xpr['D'] = cdl._xpr.numpy()[0,0,0]

        ########## CDL-E
        cdl = CDL(  "E",
                    delay_spread=TestCDL.DELAY_SPREAD,
                    carrier_frequency=TestCDL.CARRIER_FREQUENCY,
                    ut_array=rx_array,
                    bs_array=tx_array,
                    direction='downlink',
                    dtype=torch.complex128)
        a,tau = cdl(100000, 1, 100e6)
        a = a[:,0,0,0,0,:,0].numpy()
        tau = tau.numpy()[0,0,0]
        p = np.mean(np.square(np.abs(a)), axis=0)
        TestCDL.powers['E'] = p
        TestCDL.delays['E'] = tau
        TestCDL.aod['E'] = cdl._aod.numpy()[0,0,0]
        TestCDL.aoa['E'] = cdl._aoa.numpy()[0,0,0]
        TestCDL.zod['E'] = cdl._zod.numpy()[0,0,0]
        TestCDL.zoa['E'] = cdl._zoa.numpy()[0,0,0]
        TestCDL.xpr['E'] = cdl._xpr.numpy()[0,0,0]

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_powers(self, model, submodel): # Submodel does not apply to CDL
        """Test powers"""
        i = np.argsort(CDL_DELAYS[model])
        p = TestCDL.powers[model]
        ref_p = np.power(10.0, CDL_POWERS[model]/10.0)
        ref_p = ref_p/np.sum(ref_p)
        ref_p = ref_p[i]
        max_err = np.max(np.abs(ref_p - p)/ref_p)
        self.assertLessEqual(max_err, TestCDL.MAX_ERR_REL, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_delays(self, model, submodel): # Submodel does not apply to CDL
        """Test delays"""
        d = TestCDL.delays[model]/TestCDL.DELAY_SPREAD
        ref_d = np.sort(CDL_DELAYS[model])
        max_err = np.max(np.abs(ref_d - d))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_aod(self, model, submodel): # Submodel does not apply to CDL
        """Test AoD"""
        a = TestCDL.aod[model]
        ref_a = cdl_aod(model)
        max_err = np.max(np.abs(ref_a - a))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_aoa(self, model, submodel): # Submodel does not apply to CDL
        """Test AoA"""
        a = TestCDL.aoa[model]
        ref_a = cdl_aoa(model)
        max_err = np.max(np.abs(ref_a - a))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_zod(self, model, submodel): # Submodel does not apply to CDL
        """Test ZoD"""
        a = TestCDL.zod[model]
        ref_a = cdl_zod(model)
        max_err = np.max(np.abs(ref_a - a))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_zoa(self, model, submodel): # Submodel does not apply to CDL
        """Test ZoA"""
        a = TestCDL.zoa[model]
        ref_a = cdl_zoa(model)
        max_err = np.max(np.abs(ref_a - a))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

    @channel_test_on_models(('A', 'B', 'C', 'D', 'E'), ('foo',))
    def test_xpr(self, model, submodel): # Submodel does not apply to CDL
        """Test XPR"""
        a = TestCDL.xpr[model]
        ref_a = CDL_XPR[model]
        max_err = np.max(np.abs(ref_a - a))
        self.assertLessEqual(max_err, TestCDL.MAX_ERR, f'{model}')

if __name__ == '__main__':
    unittest.main() 
