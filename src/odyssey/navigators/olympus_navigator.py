from odyssey.navigators import Navigator

from olympus import Campaign, ParameterSpace, Planner, ParameterVector

from olympus.objects import (
    ObjectParameter,
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)

import torch
import copy
import pandas as pd


class OlympusNavigator(Navigator):

    requires_init_data=False

    def __init__(self, planner, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._param_space = self._parse_parameters()

        self._campaign = Campaign()
        self._campaign.set_param_space(self._param_space)

        self._planner = Planner(kind=planner, goal="maximize", param_space=self._param_space)

        self.param_inds = {}
        self.param_names = {}
        for i, param in enumerate(self.mission._params):
            name = param["name"]
            self.param_inds[name] = i
            self.param_names[i] = name

        self.objective_inds = {}
        self.objective_names = {}
        for i, objective in enumerate(self.mission._objectives):
            name = objective["name"]
            self.objective_inds[name] = i
            self.objective_names[i] = name

    def _convert_objective_vector_to_list(self, param_vec):
        return [ getattr(param_vec, name) for name in self.objective_inds ]

    def _convert_objective_list_to_vector(self, objective_data):
        return ParameterVector(
            dict={ self.objective_names[i]: data.item() for i, data in enumerate(objective_data) },
        )

    def _convert_parameters_vector_to_list(self, param_vec):
        return [ getattr(param_vec, name) for name in self.param_inds ]

    def _convert_parameters_list_to_vector(self, params_data):
        return ParameterVector(
            dict={ self.param_names[i]: data.item() for i, data in enumerate(params_data) },
            param_space=self._param_space,
        )

    def _parse_parameters(self):

        param_space = ParameterSpace()

        for data in self.mission._params:
            
            if data["type"] == "numerical":
                param = ParameterContinuous(
                    name=data["name"],
                    low=data["envelope"][0],
                    high=data["envelope"][1],
                )
            elif data["type"] == "categorical":
                param = ParameterCategorical(
                    name=data["name"],
                    options=data["categories"],
                    description=[None for _ in range(len(data["categories"]))],
                )
            elif data["type"] == "discrete":
                param = ParameterDiscrete(
                    name=data["name"],
                    low=data["low"],
                    high=data["high"],
                    stride=data["stride"],
                )
            else:
                raise NotImplementedError(f"{data['type']} parameters not supported")
            
            param_space.add(param)
        
        return param_space

    def relay(self, inputs, observations, *args, **kwargs):
        super().relay(inputs, observations, *args, **kwargs)

        for input, output in zip([inputs], [observations]):
            input_vec = self._convert_parameters_list_to_vector(input)
            
            output_vec = dict=self._convert_objective_list_to_vector(output)

            self._campaign.add_observation(
                input_vec, output_vec
            )
    
    def _upgrade(self, *args, **kwargs):
        pass

    def _get_next_trial(self, *args, **kwargs):
        params = torch.empty((0, self.mission.param_dims))
        param_vecs = self._planner.recommend(self._campaign.observations)
        if isinstance(param_vecs, list):
            for param_vec in param_vecs:
                params = torch.cat(
                    (
                        params,
                        torch.tensor(self._convert_parameters_vector_to_list(param_vec)).unsqueeze(0),
                    ), 
                )
        elif isinstance(param_vecs, ParameterVector):
            params = torch.tensor(self._convert_parameters_vector_to_list(param_vecs)).unsqueeze(0)
        
        return params