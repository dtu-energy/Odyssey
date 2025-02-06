from odyssey.navigators import Navigator

from ConfigSpace import Configuration, ConfigurationSpace

import numpy as np
import torch
from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialInfo, TrialValue



class SMACNavigator(Navigator):

    requires_init_data=False

    def __init__(self, n_trials, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._config_space = self._parse_parameters()
        self._trial_seeds = [] # currently only sequential because need to keep track of seeds, could do hash table

        self._param_names = [ x["name"] for x in self.mission._params ]
        self._objective_names = [ x["name"] for x in self.mission._objectives ]

        scenario = Scenario(
            self._config_space, 
            deterministic=kwargs.get("deterministic", False), 
            n_trials=n_trials,
        )

        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario,
            max_config_calls=1,
        )

        self._smac = HyperparameterOptimizationFacade(
            scenario,
            "",
            intensifier=intensifier,
            overwrite=True,
        )


    def _inputs_to_trial_info(self, inputs):
        values = {}
        for i, value in enumerate(inputs.squeeze()):
            values[self._param_names[i]] = value.item()
        
        return TrialInfo(
            Configuration(self._config_space, values=values), 
            seed=self._trial_seeds.pop(),
        )
    
    def _outputs_to_trial_value(self, outputs):
        
        if len(outputs) == 1:
            costs = -float(outputs.item()) # SMAC minimizes the cost hence take negative since odyssey maximises
        else:
            costs = [ -float(x.item()) for x in outputs]

        return TrialValue(cost=costs)
    
    def _trial_info_to_list(self, info):

        config = info.config
        data = []

        for name in self._param_names:
            data.append(config[name])

        return data
            

    def _parse_parameters(self, *args, **kwargs):

        param_dict = {}

        for data in self.mission._params:
            name = data["name"]
            if data["type"] == "numerical":
                param_dict[name] = tuple(data["envelope"])
            elif data["type"] == "categorical":
                param_dict[name] = data["categories"]
            else:
                raise NotImplementedError(f"{data['type']} parameters not supported")

        return ConfigurationSpace(param_dict)

    def relay(self, inputs, observations, *args, **kwargs):
        super().relay(inputs, observations, *args, **kwargs)
        # import pdb; pdb.set_trace()
        for input, output in zip([inputs], [observations]):
            self._smac.tell(
                self._inputs_to_trial_info(input),
                self._outputs_to_trial_value(output) 
            )

    def _upgrade(self, *args, **kwargs):
        pass
    
    def _get_next_trial(self, *args, **kwargs):
        info = self._smac.ask()

        self._trial_seeds.append(info.seed)

        trial = self._trial_info_to_list(info)
        
        return torch.tensor(trial).unsqueeze(0)