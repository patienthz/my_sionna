import torch

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
