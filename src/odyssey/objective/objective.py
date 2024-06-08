import torch

class Objective:
    """
    The Objective class serves as a wrapper for a function, ensuring that the output is correctly formatted as a PyTorch tensor with the appropriate dimensions.
    This is particularly beneficial when the function is utilized as an objective function in optimization tasks, such as those managed by the Navigator and Mission classes.

    Attributes:
        func (callable): The function to be wrapped. It should take a tensor as input and return a value that can be converted to a tensor.
        args (tuple): Positional arguments to be passed to the function.
        kwargs (dict): Keyword arguments to be passed to the function.

    Examples:
        >>> import torch
        >>> def real_func(x: torch.Tensor, noise_level = 0):
        ...     noise = (-1 + torch.rand(x.size()) * 2) * noise_level
        ...     return -(torch.sin(x) + torch.sin((10.0 / 3.0) * x)) + noise
        ...
        >>> objective = Objective(real_func, noise_level = 0.2)
        >>> input = torch.tensor([5, 6, 7])
        >>> objective(input)  # The function value is computed taking into account the specified noise_level parameter
        tensor([[ 1.8691],
                [-0.7979],
                [ 0.1640]])
    """


    def __init__(self, func, *args, **kwargs):

        """
        Initializes the Objective class with the given function and arguments.

        Args:
            func (callable): The function to be wrapped.
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.
        """

        self.func = func

        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs) -> torch.Tensor:

        """
        Calls the stored function with the given arguments and processes the output.
        The output is converted to a PyTorch tensor if it's not already one, and an extra dimension is added if the tensor has less than 2 dimensions.

        Args:
            *args: Positional arguments to be passed to the function.
            **kwargs: Keyword arguments to be passed to the function.

        Returns:
            torch.Tensor: The processed output of the function.
        """
        
        merged_args = (*self.args, *args)
        merged_kwargs = {**self.kwargs, **kwargs}

        output = self.func(*merged_args, **merged_kwargs)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)

        if output.dim() < 2:
            output = output.unsqueeze(-1)

        return output
