from odyssey.navigators import Navigator
import torch
from botorch.utils.sampling import draw_sobol_samples

class Sobol_Navigator(Navigator):
    requires_init_data = False

    def __init__(self,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.samples_generated = 0
    
    def _upgrade(self):
        self.samples_generated += 1

    def _trajectory(self):

        candidate = draw_sobol_samples(self.traj_bounds, n = 1, q = 1, seed = None).squeeze(0)

        return candidate
        