try:
    import my_code
except ImportError as e:
    import sys
    sys.path.append("../")
from my_code.mysionna.channel.torch_version.utils import exp_corr_mat, one_ring_corr_mat
from my_code.mysionna.channel.torch_version.spatial_correlation import KroneckerModel,PerColumnModel
from my_code.mysionna.utils import complex_normal, matrix_sqrt

import pytest
import unittest
import numpy as np
import torch

from my_code.mysionna.utils.tensors import matrix_sqrt
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

class TestKroneckerModel(unittest.TestCase):
    """Unittest for the KroneckerModel"""

    def test_covariance(self):
        M = 16
        K = 4
        dtype = torch.complex128
        r_tx = exp_corr_mat(0.4, K, dtype)
        r_rx = exp_corr_mat(0.99, M, dtype)
        batch_size = 1000000
        kron = KroneckerModel(r_tx, r_rx)

        def func():
            h = complex_normal([batch_size, M, K], dtype=dtype)
            h = kron(h)
            r_tx_hat = torch.mean(torch.matmul(h.conj().transpose(-1, -2),h), dim=0)
            r_rx_hat = torch.mean(torch.matmul(h, h.transpose(-1, -2).conj()), dim=0)
            return r_tx_hat, r_rx_hat

        r_tx_hat = torch.zeros_like(r_tx)
        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 10
        for i in range(iterations):
            tmp = func()
            r_tx_hat += tmp[0]/iterations/M
            r_rx_hat += tmp[1]/iterations/K
        self.assertTrue(np.allclose(r_tx, r_tx_hat, atol=1e-3))
        self.assertTrue(np.allclose(r_rx, r_rx_hat, atol=1e-3))

    def test_per_example_r_tx(self):
        """Configure a different tx correlation for each example"""
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 128
        r_tx = exp_corr_mat(np.random.uniform(size=[batch_size]), K, dtype)
        r_rx = exp_corr_mat(0.99, M, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], dtype=dtype)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = matrix_sqrt(r_rx) @ h[i] @ matrix_sqrt(r_tx[i])
            self.assertTrue(torch.allclose(h_corr[i], h_test))

    def test_per_example_r_rx(self):
        """Configure a different rx correlation for each example"""
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 10
        r_tx = exp_corr_mat(0.4, K, dtype)
        r_rx = exp_corr_mat(np.random.uniform(size=[batch_size]), M, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], dtype=dtype)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = matrix_sqrt(r_rx[i]) @ h[i] @ matrix_sqrt(r_tx)
            self.assertTrue(torch.allclose(h_corr[i], h_test))

    def test_per_example_corr(self):
        """Configure a different rx/tx correlation for each example"""
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 10
        r_tx = exp_corr_mat(np.random.uniform(size=[batch_size]), K, dtype)
        r_rx = exp_corr_mat(np.random.uniform(size=[batch_size]), M, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([batch_size, M, K], dtype=dtype)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = matrix_sqrt(r_rx[i]) @ h[i] @ matrix_sqrt(r_tx[i])
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_same_channel_with_different_corr(self):
        """Apply different correlation matrices to the same channel"""
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 10
        r_tx = exp_corr_mat(np.random.uniform(size=[batch_size]), K, dtype)
        r_rx = exp_corr_mat(np.random.uniform(size=[batch_size]), M, dtype)
        kron = KroneckerModel(r_tx, r_rx)
        h = complex_normal([M, K], dtype=dtype)
        h_corr = kron(h)
        for i in range(batch_size):
            h_test = matrix_sqrt(r_rx[i]) @ h @ matrix_sqrt(r_tx[i])
            self.assertTrue(np.allclose(h_corr[i], h_test))

    def test_property_setter(self):
        """Check that correlation matrices can be changed"""
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 10
        kron = KroneckerModel(None, None)

        def func():
            r_tx = exp_corr_mat(0.4, K, dtype)
            r_rx = exp_corr_mat(0.9, M, dtype)
            kron.r_tx = r_tx
            kron.r_rx = r_rx
            h = complex_normal([batch_size, M, K], dtype=dtype)
            h_corr = kron(h)
            return h, h_corr, r_tx, r_rx

        h, h_corr, r_tx, r_rx = func()
        for i in range(batch_size):
            h_test = matrix_sqrt(r_rx) @ h[i] @ matrix_sqrt(r_tx)
            self.assertTrue(np.allclose(h_corr[i], h_test, atol=1e-6))

class TestPerColumnModel(unittest.TestCase):
    def test_covariance(self):
        M = 16
        K = 4
        dtype = torch.complex128
        r_rx = one_ring_corr_mat([-45, -15, 0, 30], M, dtype=dtype)
        batch_size = 100000
        onering = PerColumnModel(r_rx)


        def func():
            h = complex_normal([batch_size, M, K], dtype=dtype)
            h = onering(h)
            h = h.permute(2, 0, 1).unsqueeze(-1)
            r_rx_hat = torch.mean(torch.matmul(h,h.conj().transpose(-2, -1)) , dim=1)
            return r_rx_hat

        r_rx_hat = torch.zeros_like(r_rx)
        iterations = 100
        for _ in range(iterations):
            r_rx_hat += func()/iterations
        print("r_rx: ", r_rx.numpy())
        print("r_rx_hat: ", r_rx_hat.numpy())
        self.assertTrue(np.allclose(r_rx.numpy(), r_rx_hat.numpy(), atol=1e-3))

    def test_per_example_corr(self):
        M = 16
        K = 4
        dtype = torch.complex64
        batch_size = 24
        r_rx = one_ring_corr_mat(np.random.uniform(size=[batch_size, K]), M, dtype=dtype)
        onering = PerColumnModel(r_rx)

        def func():
            h = complex_normal([batch_size, M, K], dtype=dtype)
            h_corr = onering(h)
            return h, h_corr

        h, h_corr = func()
        for i in range(batch_size):
            for k in range(K):
                h_test = matrix_sqrt(r_rx[i,k]) @ h[i,:,k].unsqueeze(-1)
                h_test = h_test.squeeze(-1)
                self.assertTrue(np.allclose(h_corr[i,:,k].numpy(), h_test.numpy()))

    def test_property_setter(self):
        M = 16
        K = 4
        dtype = torch.complex128
        batch_size = 24
        onering = PerColumnModel(None)
        
        def func():
            h = complex_normal([batch_size, M, K], dtype=dtype)
            r_rx = one_ring_corr_mat(torch.rand(batch_size, K) * 180 - 90, M, dtype=dtype)

            onering.r_rx = r_rx
            h_corr = onering(h)
            return h, h_corr, r_rx
        
        torch.manual_seed(1)
        h, h_corr, r_rx = func()
        for i in range(batch_size):
            for k in range(K):
                h_test = matrix_sqrt(r_rx[i,k]) @ h[i,:,k].unsqueeze(-1)
                h_test = h_test.squeeze(-1)
                self.assertTrue(np.allclose(h_corr[i,:,k].numpy(), h_test.numpy()))

if __name__ == '__main__':
    unittest.main()
