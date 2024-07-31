import torch

def scatter_nd_torch(indices, updates, shape):
    # 创建一个初始张量
    output = torch.zeros(shape, dtype=updates.dtype)
    
    # 将 indices 展开为一维，计算对应的线性索引
    flat_indices = torch.zeros(indices.shape[0], dtype=torch.long)
    stride = 1
    for i in reversed(range(indices.shape[1])):
        flat_indices += indices[:, i] * stride
        stride *= shape[i]
    
    # 使用 scatter_ 方法更新平坦输出
    for i, idx in enumerate(flat_indices):
        output[idx] = updates[i]

    return output