# Single GP Navigator Class

from typing import Type
import torch
# torch.set_default_dtype(torch.float64)

# Base Navigator
from odyssey.navigators.base_navigator import Navigator

import torch
from torch.quasirandom import SobolEngine

from botorch import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

# Aquisition Functions
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    qSimpleRegret,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement
)
from botorch.sampling.normal import SobolQMCNormalSampler

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


class BOTorch_SAASBO_Navigator(Navigator):

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
    
    _acq_funcs = {
        "expected_improvement": qLogNoisyExpectedImprovement,
        "upper_confidence_bound": qUpperConfidenceBound,
        "probability_of_improvement": qProbabilityOfImprovement,
        "simple_regret": qSimpleRegret,
    }

    _acq_func_default_params = {
        "expected_improvement": {'X_baseline': None},
        "upper_confidence_bound": {'beta': 0.1},
        "probability_of_improvement": {'best_f': None},
        "simple_regret": {},
    }

    tkwargs = {
        # "device": torch.device("mps"),
        # "dtype": torch.float32,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.double,
    }

    def __init__(self,
                 acq_function: str = "expected_improvement",
                 acq_function_params: dict = {},
                 input_scaling: bool = True, # botorch optimised for normalized inputs and standardised outputs
                 data_standardization: bool = True,
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
        
        super().__init__(
            input_scaling=input_scaling,
            data_standardization=data_standardization,
            *args, 
            **kwargs
        )

        assert len(self.mission.funcs) == 1, "SingleGPNavigator only supports single output missions"

        # Acquisition Function definition
        self.acq_function_type = acq_function
        self.acq_function_params = acq_function_params

    def _upgrade(self):

        """
        Updates the model and the acquisition function.
        """

        # TODO: Connect input_transform and outcome_transform to input_scaling and data_standardization
        self.model = SaasFullyBayesianSingleTaskGP(
            train_X=self.mission.train_X.to(**self.tkwargs),
            train_Y=self.mission.train_Y.to(**self.tkwargs)
            # outcome_transform=Standardize(m=1),
        )

        # self.mll = ExactMarginalLogLikelihood(
        #     self.model.likelihood, 
        #     self.model
        # )
        # _ = fit_gpytorch_mll(self.mll)
        
        fit_fully_bayesian_model_nuts(
            self.model,
            warmup_steps=256,
            num_samples=128,
            thinning=16,
            disable_progbar=True,
        )

        self.acq_function = self.create_acq_func()

    
    def create_acq_func(self, *args, **kwargs):

        """
        Creates the acquisition function using the specified type and parameters.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        acq_function = self._acq_funcs[self.acq_function_type]
        params = self._acq_func_default_params[self.acq_function_type]
        # import pdb; pdb.set_trace()
        if 'X_baseline' in params:
            params['X_baseline'] = self.mission.train_X.to(**self.tkwargs)
        if 'sampler' in params:
            params['sampler'] = SobolQMCNormalSampler(sample_shape=torch.Size([512]).to(**self.tkwargs))
        if 'best_f' in params:
            params['best_f'] = self.mission.train_Y.max().item().to(**self.tkwargs)

        
        params.update(self.acq_function_params)

        return acq_function(
            model=self.model,
            **params,
        )

        # qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        # self.acq_function = qLogNoisyExpectedImprovement(
        #     model=self.model,
        #     X_baseline=self.mission.train_X,
        #     sampler=qmc_sampler,
        # )
        
    def _get_next_trial(self, *args, **kwargs) -> torch.Tensor:
        
        if not hasattr(self.mission, 'train_X') or len(self.mission.train_X) < self.n_init:
            LOG.debug("Generating initial samples")
            candidate = self.init_method.get_next_trial()
            self.init_method.upgrade()
        else:
            candidate, _ = optimize_acqf(
                acq_function = self.acq_function,
                bounds = self.traj_bounds,
                q = kwargs.get('q', 1),
                num_restarts = 200,
                raw_samples = 512
            )

            # Convert input data if scaling enabled
            if self.input_scaling:
                candidate = unnormalize(candidate, self.mission.envelope)

        return candidate

        



    
    

    



    
    
    