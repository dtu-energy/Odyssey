from typing import Type
import torch
torch.set_default_dtype(torch.float64)

# Base Navigator
from odyssey.navigators import Navigator

# Acquisition Function
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.sampling.normal import SobolQMCNormalSampler

# Acquisition Function Optimization
from botorch.optim import optimize_acqf

# Model Fitting
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

# Others
# from botorch.utils.multi_objective.box_decompositions.non_dominated import (
#     FastNondominatedPartitioning,
# )

# Utils
from odyssey.utils.utils import normalize, unnormalize, standardize, unstandardize

import logging
LOG = logging.getLogger(__name__)


class BOTorch_MOO_Navigator(Navigator):

    requires_init_data = True

    def __init__(self,
            acq_function_params: dict = {},
            input_scaling: bool = True, # botorch optimised for normalized inputs and standardised outputs
            data_standardization: bool = True,
            *args, 
            **kwargs
        ):
        super().__init__(
            input_scaling=input_scaling,
            data_standardization=data_standardization,
            *args, 
            **kwargs,
        )

        assert len(self.mission.funcs) > 1, "qNEHVI_Navigator requires multiple output functions."

        # self.acq_function_type = qExpectedHypervolumeImprovement
        self.acq_function_params = acq_function_params

        assert 'ref_point' in self.acq_function_params, "Please include 'ref_point' in acq_function_params"

        # self.upgrade()

    def _upgrade(self) -> None:
        self.mll, self.model = self.create_model()
        fit_gpytorch_mll(self.mll)

        # pred = self.model.posterior(self.mission.train_X)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_function = qNoisyExpectedHypervolumeImprovement(
            model=self.model,
            X_baseline=self.mission.train_X,
            prune_baseline=True,
            sampler=sampler,
            **self.acq_function_params # includes ref_point
        )

        self.acq_function = acq_function

    def create_model(self):

        models = []
        for i in range(self.mission.train_Y.dim()):
            models.append(
                SingleTaskGP(
                    self.mission.train_X, 
                    self.mission.train_Y[:, i].unsqueeze(-1),
                )
            )

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    def _get_next_trial(self, *args, **kwargs) -> torch.Tensor:

        if not hasattr(self.mission, 'train_X') or len(self.mission.train_X) < self.n_init:
            LOG.debug("Generating initial samples")
            candidate = self.init_method.get_next_trial()
            self.init_method.upgrade()
        else:
            candidate, _ = optimize_acqf(
                acq_function=self.acq_function,
                bounds=self.traj_bounds,
                q=kwargs.get('q', 1),
                num_restarts=200,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
                sequential=True,
            )

            # Convert input data if scaling enabled
            if self.input_scaling:
                candidate = unnormalize(candidate, self.mission.envelope)
        
        return candidate