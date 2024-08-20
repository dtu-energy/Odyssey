from odyssey.navigators import Navigator

from baybe.targets import NumericalTarget
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalDiscreteParameter,
    NumericalContinuousParameter,
)
from baybe.searchspace import SearchSpace

from baybe.recommenders import (
    SequentialGreedyRecommender,
    FPSRecommender,
    TwoPhaseMetaRecommender,
    RandomRecommender,
)

from baybe import Campaign

import torch
import copy
import pandas as pd


class BaybeNavigator(Navigator):

    requires_init_data=False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._create_campaign(*args, **kwargs)

        self._param_names = [ x["name"] for x in self.mission._params ]
        self._objective_names = [ x["name"] for x in self.mission._objectives ]

    def _create_objectives(self):
        objectives = []
        for i, data in enumerate(self.mission._objectives):

            if data["type"] == "numerical":
                obj = NumericalTarget(
                    name=data["name"],
                    mode="MAX" if data["maneuver"]=="ascend" else "MIN"
                )
            else:
                raise NotImplementedError("only numerical targets supported")
            
            objectives.append(obj)

        return objectives
    
    def _create_parameters(self):

        param_list = []

        for data in self.mission._params:
            
            if data["type"] == "numerical":
                param = NumericalContinuousParameter(
                    name=data["name"],
                    bounds=data["envelope"],
                )
            elif data["type"] == "categorical":
                param = CategoricalParameter(
                    name=data["name"],
                    values=data["categories"],
                    encoding=data.get("encoding", "OHE"),
                )
            elif data["type"] == "discrete":
                param = NumericalDiscreteParameter(
                    name=data["name"],
                    values=data["values"],
                    tolerance=data.get("tolerance"),
                )
            else:
                raise NotImplementedError(f"{data['type']} parameters not supported")
            
            param_list.append(param)

        
        return param_list
    
    def _create_campaign(self, *args, **kwargs):

        objectives = self._create_objectives()
        parameters = self._create_parameters()

        if len(objectives) == 1:
            objective = SingleTargetObjective(objectives[0])
        else:
            raise NotImplementedError

        searchspace = SearchSpace.from_product(parameters)

        recommender = TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),  
            recommender=SequentialGreedyRecommender(),
            switch_after=kwargs.get("switch_after", 5),  # BOTorch singleGP
        )

        self._campaign = Campaign(searchspace, objective, recommender)

    def _list_to_dataframe(self, inputs, outputs=None):

        if outputs is None:
            columns = self._param_names
        else:
            columns = self._param_names + self._objective_names

        return pd.DataFrame(
            data=torch.cat((inputs, outputs), axis=-1),
            columns=columns
        )


    def relay(self, inputs, observations, *args, **kwargs):
        super().relay(inputs, observations, *args, **kwargs)
    
        df = self._list_to_dataframe(inputs, observations)

        self._campaign.add_measurements(df)


    def _upgrade(self, *args, **kwargs):
        pass
    
    def _get_next_trial(self, *args, **kwargs):
        
        df = self._campaign.recommend(batch_size=kwargs.get("batch_size", 1))

        return torch.tensor(df.values)