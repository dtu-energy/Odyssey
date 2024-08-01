# Single GP Navigator Class

from typing import Type
import torch
torch.set_default_dtype(torch.float64)

# Base Navigator
from odyssey.navigators.base_navigator import Navigator

# Aquisition Functions
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement, NoisyExpectedImprovement
from botorch.acquisition import ProbabilityOfImprovement, qProbabilityOfImprovement, NoisyExpectedImprovement
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.acquisition.analytic import LogExpectedImprovement, LogProbabilityOfImprovement

# Model Fitting
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from botorch.models.transforms import Normalize

# Utils
from odyssey.utils.utils import normalize, unnormalize, standardize, unstandardize

# Acquisition Function Optimization
from botorch.optim import optimize_acqf

import logging
LOG = logging.getLogger(__name__)



class SingleGP_Navigator(Navigator):

    """
    SingleGP_Navigator is a subclass of the Navigator class that uses a single Gaussian Process (GP) model for the mission.

    Attributes:
        requires_init_data (bool): A flag indicating that this navigator requires initial data.
        acq_function_type (Type): The type of acquisition function to use.
        acq_function_params (dict): The parameters for the acquisition function.
        model (SingleTaskGP): The GP model used for the mission.
        mll (ExactMarginalLogLikelihood): The marginal log likelihood of the model.
        acq_function: The acquisition function used for the mission.
    """

    requires_init_data = True
    
    acq_funcs = {
        "expected_improvement": ExpectedImprovement,
        "upper_confidence_bound": UpperConfidenceBound,
        "probability_of_improvement": ProbabilityOfImprovement,
    }

    acq_func_default_params = {
        "expected_improvement": {"best_f": 0.0},
        "upper_confidence_bound": {'beta': 0.5},
        "probability_of_improvement": {"best_f": 0.0},
    }

    def __init__(self,
                 acq_function_type: str = "expected_improvement",
                 acq_function_params: dict = {},
                 *args,
                 **kwargs
        ):

        """
        Initializes a SingleGP_Navigator object.

        Args:
            acq_function_type (Type): The type of acquisition function to use.
            acq_function_params (dict): The parameters for the acquisition function.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        
        super().__init__(*args, **kwargs)

        assert len(self.mission.funcs) == 1, "SingleGPNavigator only supports single output missions"

        # Acquisition Function definition
        self.acq_function_type = acq_function_type
        self.acq_function_params = acq_function_params

    def _upgrade(self):

        """
        Updates the model and the acquisition function.
        """

        # TODO: Connect input_transform and outcome_transform to input_scaling and data_standardization
        self.model = SingleTaskGP(self.mission.train_X, self.mission.train_Y, 
                                  #input_transform=Normalize(d=self.mission.param_dims),  
                                  outcome_transform=Standardize(m=self.mission.output_dims)
                                  )
        self.mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        _ = fit_gpytorch_mll(self.mll)

        self.create_acq_func()

    
    def create_acq_func(self, *args, **kwargs):

        """
        Creates the acquisition function using the specified type and parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # Find best_f
        # Will always be maximization, as parameters are inverted when minimizing
        
        # TODO Add Functionality for Monte Carlo Acquisition Functions
        # FIXME Change for Monte Carlo Acquisition Functions

        create_function = self.acq_funcs[self.acq_function_type]
        params = self.acq_func_default_params[self.acq_function_type]
        
        if 'best_f' in self.acq_function_params:
            self.acq_function_params['best_f'] = self.mission.train_Y.max().item()

        self.acq_function_params['maximize'] = True
        params.update(self.acq_function_params)

        self.acq_function = create_function(
            model=self.model,
            **params
        )
    
    def _get_next_trial(self, *args, **kwargs) -> torch.Tensor:
        
        if not hasattr(self.mission, 'train_X') or len(self.mission.train_X) < self.n_init:
            LOG.debug("Generating initial samples")
            candidate = self.init_method.get_next_trial()
            self.init_method.upgrade()
        else:
            candidate, _ = optimize_acqf(
                acq_function = self.acq_function,
                bounds = self.traj_bounds,
                q = 1,
                num_restarts = 200,
                raw_samples = 512
            )

        # Convert input data if scaling enabled
        if self.input_scaling:
            candidate = unnormalize(candidate, self.mission.envelope)

        return candidate

        



    
    

    



    
    
    