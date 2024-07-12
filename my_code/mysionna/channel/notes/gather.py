import torch
import numpy as np
def torch_gather(input_data, indices, axis=None, batch_dims=0):
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
    
    return out_data