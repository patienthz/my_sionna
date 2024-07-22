try:
    import my_code
except ImportError as e:
    import sys
    sys.path.append("../")

from my_code.mysionna.channel.torch_version.flat_fading_channel import GenerateFlatFadingChannel,ApplyFlatFadingChannel,FlatFadingChannel
from my_code.mysionna.channel.torch_version.utils import exp_corr_mat
from my_code.mysionna.channel.torch_version.spatial_correlation import KroneckerModel
from my_code.mysionna.utils.misc import QAMSource

import pytest
import unittest
import warnings
import numpy as np
import torch

# 获取所有可用的 GPU
gpus = torch.cuda.device_count()
print('Number of GPUs available:', gpus)

if gpus > 0:
    gpu_num = 0  # 要使用的 GPU 编号

    # 设置默认的 GPU
    torch.cuda.set_device(gpu_num)
    print('Only GPU number', gpu_num, 'used.')

    # 设置 GPU 内存增长模式
    device = torch.device(f'cuda:{gpu_num}')
    torch.cuda.empty_cache()
    try:
        torch.cuda.memory_allocated = 0
        torch.cuda.memory_reserved = 0
    except RuntimeError as e:
        print(e)

class TestGenerateFlatFading(unittest.TestCase):
    """Unittest for GenerateFlatFading"""

    def test_without_spatial_correlation(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 128
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant)
        h = gen_chn(batch_size)
        self.assertEqual(h.shape, (batch_size, num_rx_ant, num_tx_ant))
        self.assertEqual(h.dtype, torch.complex64)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, dtype=torch.complex128)
        h = gen_chn(batch_size)
        self.assertEqual(h.dtype, torch.complex128)

    def test_with_spatial_correlation(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron)

        def func():
            h = gen_chn(1000000)
            r_tx_hat = torch.mean(torch.matmul(h.transpose(-2, -1).conj(), h), dim=0)
            r_rx_hat = torch.mean(torch.matmul(h, h.transpose(-2, -1).conj()), dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / num_rx_ant
            r_rx_hat += tmp[1] / iterations / num_tx_ant
        self.assertTrue(np.allclose(r_tx.numpy(), r_tx_hat.numpy(), atol=1e-3))
        self.assertTrue(np.allclose(r_rx.numpy(), r_rx_hat.numpy(), atol=1e-3))

    def test_property_setter(self):
        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant)

        def func():
            gen_chn.spatial_corr = kron
            h = gen_chn(1000000)
            r_tx_hat = torch.mean(torch.matmul(h.transpose(-2, -1).conj(), h), dim=0)
            r_rx_hat = torch.mean(torch.matmul(h, h.transpose(-2, -1).conj()), dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0] / iterations / num_rx_ant
            r_rx_hat += tmp[1] / iterations / num_tx_ant
        self.assertTrue(np.allclose(r_tx.numpy(), r_tx_hat.numpy(), atol=1e-3))
        self.assertTrue(np.allclose(r_rx.numpy(), r_rx_hat.numpy(), atol=1e-3))

class TestGenerateApplyFading(unittest.TestCase):
    """Unittest for ApplyFlatFading"""
    def test_without_noise(self):
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        r_tx = exp_corr_mat(0.4, num_tx_ant).to(device)
        r_rx = exp_corr_mat(0.99, num_rx_ant).to(device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron).to(device)
        app_chn = ApplyFlatFadingChannel(add_awgn=False).to(device)

        h = gen_chn(batch_size).to(device)
        x = QAMSource(4)([batch_size, num_tx_ant]).to(device)
        y = app_chn([x, h])
        
        self.assertTrue(np.array_equal(y.cpu().numpy(), torch.matmul(h, x.unsqueeze(-1)).squeeze().cpu().numpy()))

    def test_with_noise(self):
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        r_tx = exp_corr_mat(0.4, num_tx_ant).to(device)
        r_rx = exp_corr_mat(0.99, num_rx_ant).to(device)
        kron = KroneckerModel(r_tx, r_rx)
        gen_chn = GenerateFlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron).to(device)
        app_chn = ApplyFlatFadingChannel(add_awgn=True).to(device)

        h = gen_chn(batch_size).to(device)
        x = QAMSource(4)([batch_size, num_tx_ant]).to(device)
        no = torch.tensor(0.1).to(device)

        y = app_chn([x, h, no])

        # Calculate the noise
        mat_result = torch.matmul(h, x.unsqueeze(-1)).squeeze(-1)
        n = y.to(device) - mat_result.to(device)

        # Calculate noise variance
        noise_var = torch.var(n).item()

        # Assert that the noise variance is approximately equal to the noise level
        self.assertAlmostEqual(no.item(), noise_var, places=3)

class TestFlatFadingChannel(unittest.TestCase):
    """Unittest for FlatFading"""

    def test_without_noise(self):
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 24
        dtype=torch.complex128
        r_tx = exp_corr_mat(0.4, num_tx_ant, dtype).to(device)
        r_rx = exp_corr_mat(0.99, num_rx_ant, dtype).to(device)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron, add_awgn=False, return_channel=True, dtype=dtype).to(device)
        x = QAMSource(4, dtype=torch.complex128)([batch_size, num_tx_ant]).to(device)
        y, h = chn(x)

        h =h.to("cpu")
        x=x.to("cpu")
        y=y.to("cpu")
        y_expend = torch.squeeze(torch.matmul(h, x.unsqueeze(-1)))
        self.assertTrue(np.array_equal(y,y_expend))
        
    def test_with_noise(self):
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 100000
        dtype=torch.complex128
        r_tx = exp_corr_mat(0.4, num_tx_ant, dtype)
        r_rx = exp_corr_mat(0.99, num_rx_ant, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, spatial_corr=kron, add_awgn=True, return_channel=True, dtype=dtype)
        x = QAMSource(4, dtype=dtype)([batch_size, num_tx_ant])
        no = torch.tensor(0.2)
        y, h = chn([x, no])

        y = y.to("cpu")
        x = x.to("cpu")
        h = h.to("cpu") 
        n = y - torch.squeeze(torch.matmul(h, x.unsqueeze(-1)))
        # Calculate noise variance
        noise_var = torch.var(n).item()

        # Assert that the noise variance is approximately equal to the noise level
        self.assertAlmostEqual(no.item(), noise_var, places=3)

    def test_no_return_channel(self):
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        batch_size = 1000000
        dtype=torch.complex64
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=False, dtype=dtype)
        x = QAMSource(4, dtype=dtype)([batch_size, num_tx_ant])
        no = torch.tensor(0.2)
        y = chn([x, no])
        y_var = torch.var(y).item()
        num_tx_ant = torch.tensor(num_tx_ant)
        self.assertAlmostEqual(y_var , (num_tx_ant + no).item(), places=2)

    def test_property_setter(self):
        torch.cuda.manual_seed(1)
        torch.manual_seed(1)
        np.random.seed(1)
        num_tx_ant = 4
        num_rx_ant = 16
        r_tx = exp_corr_mat(0.4, num_tx_ant)
        r_rx = exp_corr_mat(0.99, num_rx_ant)
        kron = KroneckerModel(r_tx, r_rx)
        chn = FlatFadingChannel(num_tx_ant, num_rx_ant, add_awgn=True, return_channel=True)
        qam_source = QAMSource(4)

        def func():
            chn.spatial_corr = kron
            x = qam_source([1000000, num_tx_ant])
            no = torch.tensor(0.2)
            y, h = chn([x, no])
            r_tx_hat = torch.mean(torch.matmul(h.transpose(-2, -1).conj(), h), dim=0)
            r_rx_hat = torch.mean(torch.matmul(h, h.transpose(-2, -1).conj()), dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/num_rx_ant
            r_rx_hat += tmp[1]/iterations/num_tx_ant
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

