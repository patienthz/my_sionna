import torch

from my_code.mysionna.utils import expand_to_rank
from my_code.mysionna import PI


def subcarrier_frequencies(num_subcarriers, subcarrier_spacing, dtype=torch.complex64):
    """
    计算“num_subcarriers”个子载波的基带频率,间隔为“subcarrier_spacing”

    参数
    ------
    num_subcarriers : int
        子载波数量

    subcarrier_spacing : float
        子载波间隔 [Hz]

    dtype : torch.dtype
        用于内部处理和输出的数据类型。
        如果提供的是复数数据类型，则使用相应的实数部分精度。
        默认为`torch.complex64` (`torch.float32`)。

    返回
    ------
    frequencies : [``num_subcarrier``], torch.float
        子载波的基带频率
    """
    if dtype in (torch.complex32,torch.complex64,torch.complex128):
        if dtype == torch.complex32:
            real_dtype = torch.float16
        elif dtype == torch.complex64:
            real_dtype = torch.float32
        elif dtype == torch.complex128:
            real_dtype = torch.float64
        else:
            raise TypeError("not find dtype type")
    elif dtype in (torch.float16,torch.float32,torch.float64):
        real_dtype = dtype
    else:
        raise AssertionError("dtype must be a complex or floating datatype")
    
    num_subcarriers_tensor = torch.tensor(num_subcarriers)  
    if torch.equal(torch.fmod(num_subcarriers_tensor,torch.tensor(2)),torch.tensor(0)):
        start = -num_subcarriers/2
        end = num_subcarriers/2
    else:
        start = -(num_subcarriers-1)/2
        end = (num_subcarriers-1)/2+1
    
    frequencies = torch.arange( start=start,
                                end=end,
                                dtype=real_dtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies
    # 测试函数
    """ num_subcarriers = 5
    subcarrier_spacing = 15e3  # 15 kHz
    frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing, dtype=torch.float32)
    print("Generated subcarrier frequencies:")
    print(frequencies)
    """

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    """
    计算给定频率下的信道频率响应。

    参数
    ------
    frequencies : [fft_size], torch.float
        计算信道响应的频率

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], torch.complex
        路径系数

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], torch.float
        路径延迟

    normalize : bool
        如果设置为`True`,则信道响应将在资源网格上归一化以确保每个资源元素的平均能量为1。默认为`False`。

    返回
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], torch.complex
        在``frequencies``下的信道频率响应
    """
    tau = torch.tensor(tau)
    real_dtype = tau.dtype

    if len(tau.shape) ==4:
        # 扩展维度以与 h 广播。添加以下维度：
        # - 接收天线数量 (2)
        # - 发射天线数量 (4)
        tau = torch.unsqueeze(torch.unsqueeze(tau, dim=2),dim=4)
        # 手动进行部分广播
        tau = tau.repeat(1,1,1,1,a.shape[4],1)
    
    # 添加时间样本维度以进行广播
    tau = torch.unsqueeze(tau, dim=6)

    # 将所有张量调整为可广播的形状
    tau = torch.unsqueeze(tau, dim=-1)
    h = torch.unsqueeze(a, dim=-1)
    frequencies = expand_to_rank(frequencies, tau.dim(),axis=0)

    # 计算所有集群抽头的傅里叶变换
    # 指数组件
    e = torch.exp(torch.complex(torch.zeros((), dtype=real_dtype),
                                -2 * PI * frequencies * tau))
    h_f = h * e
    # 对所有集群求和以获得信道频率响应
    h_f = torch.sum(h_f, dim=-3)

    if normalize:   
        # 归一化以确保每个批次示例和链接的资源网格能量为 1
        # 平均发射天线、接收天线、OFDM 符号和子载波
        c = torch.mean(torch.square(torch.abs(h_f)), 
                       dim=(2,4,5,6),
                       keepdim=True)
        c = torch.complex(torch.sqrt(c),
                          torch.zeros((),dtype=real_dtype) )
        h_f = torch.divide(h_f, c+ 1e-10)

    return h_f
    # 测试函数
    """ from my_code.mysionna.utils import GLOBAL_SEED_NUMBER
    import tensorflow as tf

    fft_size = 64
    batch_size = 2
    num_rx = 2
    num_rx_ant = 4
    num_tx = 2
    num_tx_ant = 4
    num_paths = 3
    num_time_steps = 1

    tf.random.set_seed(GLOBAL_SEED_NUMBER)

    # 生成 TensorFlow 数据
    frequencies_tf = tf.linspace(-fft_size // 2, fft_size // 2 - 1, fft_size)
    frequencies_tf = tf.cast(frequencies_tf, tf.float64)
    a_real_tf = tf.random.uniform((batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps), dtype=tf.float64)
    a_imag_tf = tf.random.uniform((batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps), dtype=tf.float64)
    a_tf = tf.complex(a_real_tf, a_imag_tf)
    tau_tf = tf.random.uniform((batch_size, num_rx, num_tx, num_paths), dtype=tf.float64)

    # 将 TensorFlow 数据转换为 NumPy 数组
    frequencies_np = frequencies_tf.numpy()
    a_np = a_tf.numpy()
    tau_np = tau_tf.numpy()


    # 将 NumPy 数组转换为 PyTorch 张量
    frequencies = torch.from_numpy(frequencies_np)
    a = torch.from_numpy(a_np)
    tau = torch.from_numpy(tau_np)

    h_f = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
    print("Generated OFDM channel frequency responses:")
    print(h_f)
    """

def cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=False):

    # tau = torch.tensor(tau)
    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = torch.unsqueeze(torch.unsqueeze(tau,dim=2),dim=4)
        # We therefore do part of it manually
        tau = tau.repeat(1,1,1,1,a.shape[4],1)
    
    # Add a time samples dimension for broadcasting
    tau = torch.unsqueeze(tau, dim=6)

    # Time lags for which to compute the channel taps
    l = torch.arange(l_min, l_max+1, dtype=real_dtype)

    # Bring tau and l to broadcastable shapes
    tau = torch.unsqueeze(tau, dim=-1)
    l = expand_to_rank(l, tau.dim(),axis=0)

    # sinc pulse shaping
    g = torch.sinc(l-tau*bandwidth)
    g = torch.complex(g, torch.tensor(0.,dtype=real_dtype))
    a = torch.unsqueeze(a, dim=-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = torch.sum(a*g, dim=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = torch.mean(torch.sum(torch.square(torch.abs(hm)),
                                dim=6, 
                                keepdim=True),
                            dim=(2,4,5),
                            keepdim=True
                       )
        c = torch.complex(torch.sqrt(c), torch.tensor(0.,dtype=real_dtype))
        hm = torch.divide(hm, c+1e-10)
    
    return hm
    #测试函数
    """ import numpy as np
    def test_cir_to_time_channel():
        #设置种子
        tf.random.set_seed(GLOBAL_SEED_NUMBER)

        # 示例参数
        bandwidth = 20e6  # 20 MHz

        # 生成模拟数据
        batch_size = 1
        num_rx = 1
        num_rx_ant = 1
        num_tx = 1
        num_tx_ant = 1
        num_paths = 5
        num_time_steps = 10

        # 随机生成路径系数（复数）
        a_tf = tf.complex(tf.random.normal([batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]),
                    tf.random.normal([batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]))

        # 随机生成路径延迟（实数）
        tau_tf = tf.random.uniform([batch_size, num_rx, num_tx, num_paths])

        #将 TensorFlow 数据转换为 NumPy 数组
        a_np = a_tf.numpy()
        tau_np = tau_tf.numpy()
        # 将 NumPy 数组转换为 PyTorch 张量
        a = torch.from_numpy(a_np)
        tau = torch.from_numpy(tau_np)
        # 时间抽头范围
        l_min = -5
        l_max = 5

        # 调用函数
        hm = cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=True)

        print(hm)
        print(hm.shape)
        print(hm.dtype)
    test_cir_to_time_channel()
    """

def time_lag_discrete_time_channel(bandwidth, maximum_delay_spread=3e-6):
    """
    Compute the smallest and largest time-lag for the discrete complex baseband channel,
    i.e., L_min and L_max.

    The smallest time-lag (L_min) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (L_max) is computed from the `bandwidth`
    and `maximum_delay_spread` as follows:

    L_max = ceil(W * tau_max) + 6

    where L_max is the largest time-lag, W the `bandwidth`,
    and tau_max the `maximum_delay_spread`.

    The default value for the `maximum_delay_spread` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    Note:
    The values of L_min and L_max computed by this function are only recommended values.
    L_min and L_max should be set according to the considered channel model. For OFDM systems,
    one also needs to be careful that the effective length of the complex baseband channel
    is not larger than the cyclic prefix length.

    Args:
        bandwidth (float): Bandwidth (W) [Hz]
        maximum_delay_spread (float): Maximum delay spread [s]. Defaults to 3us.

    Returns:
        tuple: (l_min, l_max)
            l_min (int): Smallest time-lag (L_min) for the discrete complex baseband channel. Set to -6.
            l_max (int): Largest time-lag (L_max) for the discrete complex baseband channel.
    """
    l_min = torch.tensor(-6, dtype=torch.int32)
    l_max = torch.ceil(torch.tensor(maximum_delay_spread * bandwidth, dtype=torch.float32)) + 6
    l_max = torch.tensor(l_max, dtype=torch.int32)
    return l_min, l_max

""" # 测试函数
bandwidth = 1e6  # 1 MHz
max_delay_spread = 3e-6  # 3 microseconds
l_min, l_max = time_lag_discrete_time_channel(bandwidth, max_delay_spread)
print(f"l_min: {l_min}, l_max: {l_max}") """




