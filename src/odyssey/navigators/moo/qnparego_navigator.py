from typing import Type
import torch
torch.set_default_dtype(torch.float64)

# Base Navigator
from odyssey.navigators import Navigator

# Acquisition Functions
from botorch.acquisition import qNoisyExpectedImprovement

# Model Fitting
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls import SumMarginalLogLikelihood
from botorch import fit_gpytorch_mll

# Acquisition Function Optimization
from botorch.optim.optimize import optimize_acqf_list

# Others
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

class qNParEGO_Navigator(Navigator):

    requires_init_data = True

    def __init__(self, acq_function_params: dict, *args, **kwargs):
        
        super().__init__(*args, **kwargs)

        assert len(self.mission.funcs) > 1, "qNParEGO_Navigator requires multiple output functions. For single output functions, use SingleGPNavigator"

        self.acq_function_type = qNoisyExpectedImprovement
        self.acq_function_params = acq_function_params

        self.upgrade()

    def _upgrade(self):
  
        self.mll, self.model = self.initialize_model()
        fit_gpytorch_mll(self.mll)

        pred = self.model.posterior(self.mission.train_X)

        n_candidates = 1
        acq_fun_list = []
        for _ in range(n_candidates):
            weights = sample_simplex(self.mission.train_Y.dim()).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(
                    weights,
                    pred.mean
                )
            )

            expected_improvement = self.acq_function_type(
                model=self.model,
                objective=objective,
                X_baseline=self.mission.train_X,
                **self.acq_function_params
            )

            acq_fun_list.append(expected_improvement)

        self.acq_function = acq_fun_list

    def initialize_model(self):

        models = []
        for i in range(self.mission.train_Y.dim()):
            train_objective = self.mission.train_Y[:, i]
            models.append(
                SingleTaskGP(self.mission.train_X, train_objective.unsqueeze(-1))
            )

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        return mll, model

    def _trajectory(self):

        candidate, _ = optimize_acqf_list(
            acq_function_list=self.acq_function,
            bounds=self.traj_bounds,
            num_restarts=200,
            raw_samples=512
        )

        return candidate

