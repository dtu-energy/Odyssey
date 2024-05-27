from odyssey.navigators import Navigator
import torch

class Grid_Navigator(Navigator):
    requires_init_data = False

    def __init__(self,
                 subdivisions: int,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        

        self.iter_value = 0
        
        tensor_dims = [torch.randperm(subdivisions) * (upper - lower) / (subdivisions - 1) + lower for lower, upper in self.traj_bounds.T]
        self.x = torch.stack(tensor_dims, dim=1)

        # TODO Add some kind of stop so that we don't go over subdivisions

    def _upgrade(self):
        self.iter_value += 1

    def _trajectory(self):
        candidate = self.x[[self.iter_value]]

        return candidate