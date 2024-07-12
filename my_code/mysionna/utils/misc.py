import torch

def complex_normal(shape, var=1.0, dtype=torch.complex64):
    # Half the variance for each dimension
    if dtype == torch.complex32:
        real_dtype =  torch.float16
    if dtype == torch.complex64:
        real_dtype = torch.float32
    if dtype == torch.complex128:
        real_dtype = torch.float64

    
    var_dim = torch.tensor(var,dtype=real_dtype)/torch.tensor(2,dtype=real_dtype)
    stddev = torch.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = torch.normal(mean=0,std=stddev, size=shape, dtype= real_dtype)
    xi = torch.normal(mean=0,std=stddev, size=shape, dtype= real_dtype)
    x = torch.view_as_complex(torch.stack((xr, xi), axis=-1))

    return x
