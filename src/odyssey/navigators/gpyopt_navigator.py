from odyssey.navigators import Navigator

import GPyOpt
from GPyOpt.methods import BayesianOptimization

import torch
import copy
import pandas as pd


class GPyOptNavigator(Navigator):

    requires_init_data=False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert len(self.mission._objectives) == 1, "GPyOpt cannot have more than one objective"

        self.kwargs = kwargs
        
    def _get_model(self):
        kwargs = copy.deepcopy(self.kwargs)

        return BayesianOptimization(
            f=lambda x: x,
            domain=self.domain,
            X=self.mission.train_X.numpy() if len(self.mission.train_X) > 0 else None,
            Y=self.mission.train_Y.numpy() if len(self.mission.train_Y) > 0 else None,
            normalize_Y=kwargs.pop("normalize_Y", False),
            maximize=kwargs.pop("maximize", True),
            **kwargs,
        )
    

    @property
    def domain(self):

        if hasattr(self, "_domain"):
            return getattr(self, "_domain")
        
        domain = []
        for data in self.mission._params:
            if data["type"]=="numerical":
                data["type"] = "continuous"
                data["domain"] = data.pop("envelope")
            elif data["type"]=="discrerte":
                data["domain"] = data.pop("values")
            else:
                raise NotImplementedError(f"parameter {data['type']} not able to be parsed")
        
            domain.append(data)
        
        self._domain = domain

        return domain

    def _upgrade(self, *args, **kwargs):
        pass

    def _get_next_trial(self, *args, **kwargs):
        model = self._get_model()
        arr = model.suggest_next_locations()

        return torch.tensor(arr)

