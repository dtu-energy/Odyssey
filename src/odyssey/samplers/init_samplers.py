import torch

def init_grid(bounds, num_points):
   
    """
    Generate a torch tensor with specified number of dimensions, bounds for each dimension, and number of data points.

    Parameters:
    - bounds (list of tuples): List of tuples, where each tuple represents the lower and upper bounds for a dimension.
    - num_points (int): Number of data points for each dimension. 

    Returns:
    - torch.Tensor: Generated tensor.
    """


    tensor_dims = [torch.randperm(num_points) * (upper - lower) / (num_points - 1) + lower for lower, upper in bounds]
    x = torch.stack(tensor_dims, dim=1)

    return x



