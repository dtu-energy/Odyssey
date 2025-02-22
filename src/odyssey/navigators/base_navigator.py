### Base Navigator Class ###

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
torch.set_default_dtype(torch.float64)

# Mission
from odyssey.mission import Mission

# Utils
from odyssey.utils.utils import normalize, unnormalize, standardize, unstandardize

class Navigator(ABC):

    """
    Base Navigator class for the Odyssey API. This class is an abstract base class (ABC) that defines the basic structure 
    for all Navigator classes in the Odyssey API. It provides methods for generating initial data, display data, log data, 
    and for updating the model and training data.
    """

    requires_init_data = True

    def __init__(
            self,
            mission: Mission,
            n_init: int = 5,
            init_method: str = "sobol",
            input_scaling: bool = False,
            data_standardization: bool = False,
            display_always_max: bool = False,
            init_kwargs: dict = {},
        ):

        """
        Initializes a Navigator object.

        Args:
            mission (Mission): The mission object associated with the Navigator.
            n_init (int, optional): The number of initial datapoints. Defaults to None.
            init_method (Navigator, optional): The method used for initialization. See Sampler Navigators for more information. Defaults to None.
            input_scaling (bool, optional): Specifies if input parameters are normalized to the unit cube for model training. Defaults to False.
            data_standardization (bool, optional): Specifies if output parameters are standadized (zero mean and unit variance) for model training. Defaults to False.
            display_always_max (bool, optional): If set to true, minimization problems are logged and displayed as maximization problems. Useful for users who prefer viewing all problems as maximization tasks. Defaults to False.
        """

        self.mission = mission
    
        # Check if Navigator requires init data
        ## Sampler-Type Navigators do not require init data
        ## Acquisition-Type Navigators require init data

        

        # FIXME Input scaling not working correctly.

        self.input_scaling = input_scaling
        self.data_standardization = data_standardization
        self.display_always_max = display_always_max

        if self.input_scaling:
            self.traj_bounds = torch.stack(
                (torch.zeros(self.mission.param_dims), torch.ones(self.mission.param_dims))
            ) # Unit Cube bounds
        else:
            self.traj_bounds = self.mission.envelope.T

        if self.requires_init_data:
            assert n_init > 0, "This navigator requires initial data, but n_init is set to 0"
            self.n_init = n_init
            self.init_method = self.get_init_navigator(
                init_method=init_method,
                n_init=n_init,
                **init_kwargs
            )
        

        self.mission.train_X = torch.empty((0, mission.param_dims))
        self.mission.train_Y = torch.empty((0, mission.output_dims))

        self.mission.display_X = self.mission.train_X.clone()
        self.mission.display_Y = self.mission.train_Y.clone()


    def get_init_navigator(self, init_method, n_init, **kwargs):
        """
        Set up the initial navigator for generating datapoints before optimising

        Args:
            init_method (str): the name of the initial navigator
        """
        from odyssey.navigators.sampler_navigators import (
            Sobol_Navigator,
            Grid_Navigator,
            Random_Navigator,
        )

        initial_navigators = {
            "sobol": Sobol_Navigator,
            "grid": Grid_Navigator,
            "random": Random_Navigator,
        }

        return initial_navigators[init_method](
            n_init,
            mission=self.mission,
            **kwargs,
        )

    
    def generate_display_data(self):

        """
        Converts the training data to display data.
        """

        if self.input_scaling:
            display_input = unnormalize(self.mission.train_X.clone(), self.mission.envelope)
        else:
            display_input = self.mission.train_X.clone()

        if self.data_standardization:
            display_output = unstandardize(self.mission.train_Y.clone(), 
                                           mean = self.init_train_Y_mean, 
                                           std = self.init_train_Y_std
                                           )
        else:
            display_output = self.mission.train_Y.clone()

            
        if self.display_always_max == False:
            # Minimization (descend) objectives were inverted during optimization
            # Values of these objectives are now inverted back for displaying if display_always_max is False
            # This way, the user can see the actual values of the objectives and not of the forced maximization
            descend_indices = [idx for idx, value in enumerate(self.mission.maneuvers) if value == 'descend']
            display_output[:, descend_indices] *= -1

        return display_input, display_output
    
    def generate_log_data(self, init: bool = False) -> dict:

        """
        Generates log data for the mission.

        Args:
            init (bool, optional): Specifies if this is the initial log data generation. Defaults to False.
        """
        
        if init:
            display_X_subset = self.mission.display_X
            display_Y_subset = self.mission.display_Y
        else:
            display_X_subset = self.mission.display_X[-1]
            display_Y_subset = self.mission.display_Y[-1]

        if display_X_subset.dim() == 1 or display_Y_subset.dim() == 1:
            display_X_subset = display_X_subset.unsqueeze(0)
            display_Y_subset = display_Y_subset.unsqueeze(0)

        trajectory_dict = {f'{self.mission._params[i]["name"]}': display_X_subset[:, i].tolist() for i in range(display_X_subset.shape[1])}
        observation_dict = {f'{self.mission._objectives[i]["name"]}': display_Y_subset[:, i].tolist() for i in range(display_Y_subset.shape[1])}

        data_dict = {**trajectory_dict, **observation_dict}

        return data_dict

    @abstractmethod
    def _upgrade(self, *args, **kwargs):

        """
        Abstract method to update the model with specific requirements.
        """

        pass

    def upgrade(self, *args, **kwargs):
        
        """
        Updates the model by calling the model-specific abstract `_upgrade` method.
        """

        self._upgrade()
    
    @abstractmethod
    def _get_next_trial(self, *args, **kwargs):
        """
        Abstract method to translate model-specific parameter recommendation to a compatible format (torch.tensor).
        """

        pass
    
    def get_next_trial(self, *args, **kwargs):
        """
        Translates model-specific parameter recommendation to a compatible format (torch.tensor) by calling the abstract `_trajectory` method.
        """

        return self._get_next_trial(*args, **kwargs)

    def relay(self, inputs, observations, *args, **kwargs):

        """
        Updates the training data and display data with the new inputs and observation. 

        Args:
            inputs: The trial inputs associated with the observations.
            observations: The new observation.
        """
        # relay raw inputs and outputs
        self.mission.display_X = torch.cat((self.mission.display_X, inputs))
        self.mission.display_Y = torch.cat((self.mission.display_Y, observations))

        descend_indices = [idx for idx, value in enumerate(self.mission.maneuvers) if value == 'descend']
        observations[:, descend_indices] *= -1

        # normalize and relay training input data
        if self.input_scaling:
            inputs = normalize(inputs, self.mission.envelope)
        self.mission.train_X = torch.cat((self.mission.train_X, inputs))

        # standardise and relay training output data
        if self.requires_init_data and self.data_standardization and len(self.mission.display_Y) >= self.n_init:
            self.mission.train_Y = self.mission.display_Y.clone()
            self.mission.train_Y = standardize(
                self.mission.train_Y,
                mean = self.mission.train_Y.mean(dim=0),
                std = self.mission.display_Y.std(dim=0),
            )
        else:
            self.mission.train_Y = torch.cat((self.mission.train_Y, observations))

        # Log data
        ## Case where only one trajectory point and one observation point is observed (q=1)
        if self.mission.log_data:
            data_dict = self.generate_log_data(init = False)
            self.mission.write_to_logfile(data = data_dict)
        
    
    def probe(self, input_data: torch.tensor, init: bool, *args, **kwargs) -> torch.Tensor:

        """
        Probes the functions with the input data and returns the output.

        Args:
            input_data (torch.tensor): The input data to probe the functions with.
            init (bool): A flag indicating whether this is the initial probing.

        Returns:
            torch.Tensor: The output from all functions.
        """

        for f in range(len(self.mission.funcs)):
            
            for d in range(len(input_data)):
                data = input_data[d].unsqueeze(0)
                probed_value = self.mission.funcs[f](data)

                if d == 0:
                    output = probed_value
                else:
                    output = torch.cat((output, probed_value), dim = 0)
                
            # Ensure > 1D output
            if output.dim() < 2:
                output = output.unsqueeze(-1)
            
            # Concatenate multiple outputs
            if f == 0:
                output_all = output
            else:
                output_all = torch.cat((output_all, output), dim=1)

        return output_all

                
            
       


