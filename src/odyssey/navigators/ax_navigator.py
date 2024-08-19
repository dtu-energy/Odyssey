from odyssey.navigators import Navigator
from ax.service.ax_client import AxClient, ObjectiveProperties

import torch
import copy


class AxClientNavigator(Navigator):

    requires_init_data=False

    def __init__(self, exp_kwargs={}, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._exp_kwargs = exp_kwargs
        self._hashed_trials = {}

        self._client = AxClient()

        self._create_experiment()

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

    def _convert_objective_dict_to_list(self, data):
        return [ data[name] for name in self.objective_inds ]

    def _convert_objective_list_to_dict(self, objective_data):
        return { self.objective_names[i]: data.item() for i, data in enumerate(objective_data) }

    def _convert_parameters_dict_to_list(self, params):
        return [ params[name] for name in self.param_inds ]

    def _convert_parameters_list_to_dict(self, params_data):
        return { self.param_names[i]: data for i, data in enumerate(params_data) }

    def _create_parameter(self, param_data):
        param_dict = copy.deepcopy(param_data)

        if param_dict["type"] == "range":
            param_dict["bounds"] = param_dict.pop("envelope")
        elif param_dict["type"] == "categorical":
            param_dict["values"] = param_dict.pop("categories")
        
        return param_dict
    
    def _create_objectives(self, objective_data):
        objectives = {}

        for data in objective_data:
            name = data["name"]
            minimize = True if data["maneuver"]=="descend" else False
            objectives[name] = ObjectiveProperties(minimize=minimize)

        return objectives

    def _create_experiment(self):
        
        self._client.create_experiment(
            name=self.mission.name,
            parameters=[
                self._create_parameter(x) for x in self.mission._params
            ],
            objectives=self._create_objectives(self.mission._objectives),
            **self._exp_kwargs,
        )

    def _get_next_trial(self, *args, **kwargs):

        if kwargs.get('q', 1):
            parameterizations, trial_number = self._client.get_next_trial(
                *args,
                **kwargs,
            )

            param_list = torch.tensor(self._convert_parameters_dict_to_list(parameterizations)).unsqueeze(0)

            self._hashed_trials[hash(str(param_list))] = trial_number # this is probably silly, look at human in loop tutorial for potentially better way

            return param_list
        else:
            parameterizations, _ = self._client.get_next_trials(
                max_trials=kwargs.get('q'),
                *args,
                **kwargs,
            )

            self._hashed_trials.update({
                hash(str(self._convert_parameters_dict_to_list(v))): k for k, v in parameterizations.items()
            })

            return torch.tensor([ self._convert_parameters_dict_to_list(x) for x in parameterizations.values() ]).unsqueeze(0)

    def relay(self, inputs, observations, *args, **kwargs):
        super().relay(inputs, observations, *args, **kwargs)

        for input, observation in zip([inputs], [observations]):
            trial_number = self._hashed_trials[hash(str(input))]

            self._client.complete_trial(
                trial_number,
                self._convert_objective_list_to_dict(observation)
            )

    def _upgrade(self):
        pass

 