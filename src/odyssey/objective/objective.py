import torch

class Objective:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):
        output = self.func(*args, **kwargs)
        if not isinstance(output, torch.Tensor):
            output = torch.tensor(output)

        if output.dim() < 2:
            output = output.unsqueeze(-1)

        return output
