import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import torch

# 参数说明：
# seed --> 随机种子
# shape --> 生成形状
# dtype --> 数据类型 
# version --> 返回类型 0 代表 tf ； 1 代表 torch
# 功能说明，随机生成 numpy 类型随机数 ， 并转换成 tf.tensor or Tensor（torch）
def set_random_numbers(seed=100,shape=None,dtype=np.float32,version=None):
    np.random.seed(seed)
    np_random_numbers = np.random.rand(shape)
    #version =0 :tensorflow;version =1 :pytorch
    if version == 0:
        tf_random_numbers = tf.convert_to_tensor(np_random_numbers, dtype=tf.float32)
        return tf_random_numbers
    elif version == 1:
        torch_random_numbers=torch.tensor(np_random_numbers,dtype=torch.float32)
        return torch_random_numbers
    

def torch_rand_uniform(seed, shape, dtype, device):
    tf.random.set_seed(seed)
    data1 = tf.random.uniform(shape, dtype, )
    data2 = torch.tensor(data1.numpy()).to(device)
    return data2