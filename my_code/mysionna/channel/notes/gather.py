import torch
import numpy as np
""" def torch_gather(input_data, indices, axis=None, batch_dims=0):
    shape_input = input_data.shape
    shape_index = indices.shape

    shape_new = np.array(shape_input)
    shape_new = np.delete(shape_new, axis)
    shape_new = np.insert(shape_input, axis, shape_index)

    data_new = torch.tensor(shape_new)
    
    for i in range(len(shape_index) - 1):
        cu_index =  len(shape_index) + axis - i - 1
        torch.index_select(input_data, axis, indices)
    else:
        out_data = torch.index_select(input_data, axis, indices)
    
    return out_data """

def gather_pytorch(input_data,  axis=0, indices=None):
    if axis < 0:
        axis = len(input_data.shape) + axis
        
    data = torch.index_select(input_data, axis, indices.flatten())

    shape_input = list(input_data.shape)
    # shape_ = delete(shape_input, axis)
    
    # 连接列表
    shape_output = shape_input[:axis] + \
        list(indices.shape) + shape_input[axis + 1:]

    data_output = data.reshape(shape_output)

    return data_output