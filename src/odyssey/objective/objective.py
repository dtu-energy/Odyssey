import torch

class Objective:
    def __init__(self, func, *args, **kwargs):
        self.func = func

        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, *args, **kwargs):
        
        merged_args = (*self.args, *args)
        merged_kwargs = {**self.kwargs, **kwargs}

        output = self.func(*merged_args, **merged_kwargs)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)

        if output.dim() < 2:
            output = output.unsqueeze(-1)

        return output
