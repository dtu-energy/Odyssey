from odyssey.navigators import Navigator

from bofire.data_models.features.api import (
    ContinuousInput,
    DiscreteInput,
    CategoricalInput,
    CategoricalDescriptorInput,
)
from bofire.data_models.features.api import ContinuousOutput
from bofire.data_models.objectives.api import MaximizeObjective, MinimizeObjective
from bofire.data_models.domain.api import Inputs, Outputs
from bofire.data_models.strategies.api import RandomStrategy
import bofire.strategies.api as strategies
from bofire.data_models.domain.api import Domain
from bofire.data_models.strategies.api import SoboStrategy, MoboStrategy
from bofire.data_models.acquisition_functions.api import qLogNEI, qLogNEHVI

import torch
import copy
import pandas as pd


class BofireNavigator(Navigator):

    requires_init_data=True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._param_names = [ x["name"] for x in self.mission._params ]
        self._objective_names = [ x["name"] for x in self.mission._objectives ]

        self._domain = self._create_domain(*args, **kwargs)

        if len(self._domain.outputs) > 1:
            self._strategy_model = MoboStrategy(
                domain=self._domain,
                acquisition_function=kwargs.get("acquisition_function", qLogNEHVI()),
            )
        else:
            self._strategy_model = SoboStrategy(
                domain=self._domain,
                acquisition_function=kwargs.get("acquisition_function", qLogNEI()),
            )
        
        self._strategy = strategies.map(self._strategy_model)

    def _create_domain(self, *args, **kwargs):
        inputs=[]
        outputs=[]
        for data in self.mission._params:
            
            if data["type"] == "numerical":
                param = ContinuousInput(
                    key=data["name"],
                    bounds=tuple(data["envelope"]),
                )
            elif data["type"] == "categorical":
                param = CategoricalInput(
                    key=data["name"],
                    categories=data["categories"],
                )
            elif data["type"] == "discrete":
                param = DiscreteInput(
                    key=data["name"],
                    values=data["values"],
                )
            else:
                raise NotImplementedError(f"{data['type']} parameters not supported")
            
            inputs.append(param)

        for data in self.mission._objectives:
            
            if data["maneuver"]=="ascend":
                objective=MaximizeObjective(
                    w=data.get('w', 1.0),
                    bounds=data.get('bounds', (0.0, 1.0))
                )
            else:
                objective=MinimizeObjective(
                    w=data.get('w', 1.0),
                    bounds=data.get('bounds', (0.0, 1.0))
                )

            outputs.append(ContinuousOutput(
                key=data["name"],
                objective=objective
            ))
        
        return Domain.from_lists(inputs=inputs, outputs=outputs)
    
    def _list_to_dataframe(self, inputs, outputs=None):

        if outputs is None:
            data = inputs
            columns = self._param_names
        else:
            data = torch.cat((inputs, outputs), axis=-1)
            columns = self._param_names + self._objective_names # assuming output values ordered as objectives

        return pd.DataFrame(
            data=data,
            columns=columns
        )

    def _upgrade(self, *args, **kwargs):
        return super()._upgrade(*args, **kwargs)
    
    def relay(self, inputs, observations, *args, **kwargs):
        super().relay(inputs, observations, *args, **kwargs)

        df = self._list_to_dataframe(inputs, observations)

        self._strategy.tell(df)

    def _get_next_trial(self, *args, **kwargs):
        df = self._strategy.ask(
            candidate_count=kwargs.pop('q', 1),
            **kwargs,
        )

        return torch.tensor(df.iloc[:, :len(self._domain.inputs)].values) # slice to remove predicted y values

