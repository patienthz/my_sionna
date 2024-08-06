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


from my_code.mysionna.channel.torch_version.apply_ofdm_channel import ApplyOFDMChannel
from my_code.mysionna.channel.torch_version.apply_time_channel import ApplyTimeChannel

import pytest
import unittest
import numpy as np
import torch


class TestApplyTimeChannel(unittest.TestCase):

    def test_apply_time_channel(self):
        batch_size = 16
        num_rx = 4
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 2
        NUM_TIME_SAMPLES = [1, 5, 32, 128]
        L_TOT = [1, 3, 8, 16]
        for num_time_samples in NUM_TIME_SAMPLES:
            for l_tot in L_TOT:
                apply = ApplyTimeChannel(num_time_samples, l_tot, False)
                x = torch.randn(size=[batch_size,
                                      num_tx,
                                      num_tx_ant,
                                      num_time_samples])
                h_time = torch.randn(size=[batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_tx,
                                           num_tx_ant,
                                           num_time_samples+l_tot-1,
                                           l_tot])
                y = apply((x, h_time)).numpy()
                self.assertEqual(y.shape, (batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_time_samples+l_tot-1))
                y_ref = np.zeros([batch_size,
                                  num_rx,
                                  num_rx_ant,
                                  num_time_samples+l_tot-1], dtype=np.complex64)
                h_time = h_time.numpy()
                x = x.numpy()
                for b in np.arange(batch_size):
                    for rx in np.arange(num_rx):
                        for ra in np.arange(num_rx_ant):
                            for t in np.arange(num_time_samples+l_tot-1):
                                h_ = h_time[b,rx,ra,:,:,t,:]
                                x_ = x[b]
                                for l in np.arange(l_tot):
                                    if t-l < 0:
                                        break
                                    if t-l > num_time_samples-1:
                                        continue
                                    y_ref[b,rx,ra,t] += np.sum(x_[:,:,t-l]*h_[:,:,l])
                self.assertTrue(np.allclose(y_ref, y, atol=1e-5))


class TestApplyOFDMChannel(unittest.TestCase):

    def test_apply_ofdm_channel(self):
        batch_size = 16
        num_rx = 4
        num_rx_ant = 4
        num_tx = 2
        num_tx_ant = 2
        NUM_OFDM_SYMBOLS = [1, 14, 28, 64]
        FFT_SIZE = [1, 12, 32, 64]
        apply = ApplyOFDMChannel(False)
        for num_ofdm_symbols in NUM_OFDM_SYMBOLS:
            for fft_size in FFT_SIZE:
                x = torch.randn([batch_size,
                                      num_tx,
                                      num_tx_ant,
                                      num_ofdm_symbols,
                                      fft_size])
                h_freq = torch.randn([batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_tx,
                                           num_tx_ant,
                                           num_ofdm_symbols,
                                           fft_size])
                y = apply((x, h_freq)).numpy()
                self.assertEqual(y.shape, (batch_size,
                                           num_rx,
                                           num_rx_ant,
                                           num_ofdm_symbols,
                                           fft_size))
                y_ref = np.zeros([batch_size,
                                  num_rx,
                                  num_rx_ant,
                                  num_ofdm_symbols,
                                  fft_size], dtype=np.complex64)
                h_freq = h_freq.numpy()
                x = x.numpy()
                for b in np.arange(batch_size):
                    for rx in np.arange(num_rx):
                        for ra in np.arange(num_rx_ant):
                            h_ = h_freq[b,rx,ra]
                            x_ = x[b]
                            y_ref[b,rx,ra] += np.sum(x_*h_, axis=(0,1))
                self.assertTrue(np.allclose(y_ref, y, atol=1e-5))
if __name__ == "__main__":
    unittest.main()
