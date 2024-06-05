# Single GP Navigator Class

from typing import Type
import torch
torch.set_default_dtype(torch.float64)

# Base Navigator
from odyssey.navigators.base_navigator import Navigator

# Model Fitting
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

# Acquisition Function Optimization
from botorch.optim import optimize_acqf


class SingleGP_Navigator(Navigator):
    requires_init_data = True

    def __init__(self,
                 acq_function_type: Type,
                 acq_function_params: dict,
                 *args,
                 **kwargs
        ):
        
        super().__init__(*args, **kwargs)

        assert len(self.mission.funcs) == 1, "SingleGPNavigator only supports single output missions"

        # Acquisition Function definition
        self.acq_function_type = acq_function_type
        self.acq_function_params = acq_function_params

        # Create model and acquisition function
        self.upgrade()


    def _upgrade(self):
        self.model = SingleTaskGP(self.mission.train_X, self.mission.train_Y)
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _ = fit_gpytorch_mll(self.mll)

        self.create_acq_func()

    
    def create_acq_func(self, *args, **kwargs):
        # Find best_f
        # Will always be maximization, as parameters are inverted when minimizing
        
        # TODO Add Functionality for Monte Carlo Acquisition Functions
        # FIXME Change for Monte Carlo Acquisition Functions
        
        if 'best_f' in self.acq_function_params:
            self.acq_function_params['best_f'] = self.mission.train_Y.max().item()

        self.acq_function = self.acq_function_type(
                            model=self.model,
                            **self.acq_function_params             
                        )
        
    def _trajectory(self):

        candidate, _ = optimize_acqf(
            acq_function = self.acq_function,
            bounds = self.traj_bounds,
            q = 1,
            num_restarts = 200,
            raw_samples = 512
        )

        return candidate

        



    
    

    



    
    
    