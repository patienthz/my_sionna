import torch
import my_code.mysionna as sn

def expand_to_rank(tensor, target_rank, axis=-1):
    """
    Inserts as many axes to a tensor as needed to achieve a desired rank.

    This operation inserts additional dimensions to a tensor starting at
    `axis`, so that the rank of the resulting tensor has rank `target_rank`.
    The dimension index follows Python indexing rules, i.e., zero-based,
    where a negative index is counted backward from the end.

    Args:
        tensor (torch.Tensor): A tensor.
        target_rank (int): The rank of the output tensor.
            If `target_rank` is smaller than the rank of `tensor`,
            the function does nothing.
        axis (int): The dimension index at which to expand the
               shape of `tensor`. Given a `tensor` of `D` dimensions,
               `axis` must be within the range `[-(D+1), D]` (inclusive).

    Returns:
        torch.Tensor: A tensor with the same data as `tensor`, with
            `target_rank` - rank(`tensor`) additional dimensions inserted at the
            index specified by `axis`.
            If `target_rank` <= rank(`tensor`), `tensor` is returned.
    """
    current_rank = len(tensor.shape)
    num_dims = max(target_rank - current_rank, 0)
    for _ in range(num_dims):
        tensor = torch.unsqueeze(tensor, dim=axis)
    return tensor

def insert_dims(tensor, num_dims, axis=-1):
    """Adds multiple length-one dimensions to a tensor.

    This operation inserts `num_dims` dimensions of length one starting from the
    dimension `axis` of a `tensor`. The dimension index follows Python indexing rules,
    i.e., zero-based, where a negative index is counted backward from the end.

    Args:
        tensor (torch.Tensor): A PyTorch tensor.
        num_dims (int): The number of dimensions to add.
        axis (int, optional): The dimension index at which to expand the
                              shape of `tensor`. Default is -1.

    Returns:
        torch.Tensor: A tensor with the same data as `tensor`, with `num_dims`
                      additional dimensions inserted at the index specified by `axis`.
    """
    assert num_dims >= 0, "`num_dims` must be nonnegative."

    rank = tensor.dim()
    assert -(rank + 1) <= axis <= rank, "`axis` is out of range `[-(D+1), D]`."

    axis = axis if axis >= 0 else rank + axis + 1
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [1] * num_dims + shape[axis:]
    output = tensor.view(new_shape)

    return output

def matrix_sqrt(tensor):
    r""" Computes the square root of a matrix.

    Given a batch of Hermitian positive semi-definite matrices
    :math:`\mathbf{A}`, returns matrices :math:`\mathbf{B}`,
    such that :math:`\mathbf{B}\mathbf{B}^H = \mathbf{A}`.

    The two inner dimensions are assumed to correspond to the matrix rows
    and columns, respectively.

    Args:
        tensor ([..., M, M]) : A tensor of rank greater than or equal
            to two.

    Returns:
        A tensor of the same shape and type as ``tensor`` containing
        the matrix square root of its last two dimensions.
    """
    if sn.config.xla_compat and not tensor.is_grad_enabled():
        s, u = torch.linalg.eigh(tensor)

        # Compute sqrt of eigenvalues
        s = torch.abs(s)
        s = torch.sqrt(s)
        s = s.type(dtype=u.dtype)

        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, torch.conj(torch.transpose(u, -2, -1)))
    else:
        s, u = torch.linalg.eigh(tensor)

        # Compute sqrt of eigenvalues
        s = torch.abs(s)
        s = torch.sqrt(s)
        s = s.type(dtype=u.dtype)

        # Matrix multiplication
        s = s.unsqueeze(-2)
        return torch.matmul(u * s, torch.conj(torch.transpose(u, -2, -1)))