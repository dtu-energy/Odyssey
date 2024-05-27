### Base Navigator Class ###

from abc import ABC, abstractmethod

import torch
torch.set_default_dtype(torch.float64)

# Mission
from odyssey.mission import Mission

# Utils
from odyssey.utils.utils import normalize, unnormalize, standardize, unstandardize

# Init Samplers
from odyssey.samplers import init_grid

class Navigator(ABC):
    
    def __init__(self,
                 mission: Mission, 
                 num_init_design: int,
                 init_method: str,
                 input_scaling: bool,
                 data_standardization: bool,
                 display_always_max: bool = False,
        ):

        # FIXME Input scaling not working correctly.

        self.mission = mission
        self.num_init_design = num_init_design
        self.init_method = init_method

        self.input_scaling = input_scaling
        self.data_standardization = data_standardization
        self.display_always_max = display_always_max

        if self.input_scaling:
            self.traj_bounds = torch.stack((torch.zeros(self.mission.param_dims), torch.ones(self.mission.param_dims))) # Unit Cube bounds
        else:
            self.traj_bounds = self.mission.envelope.T

        

        # TODO Route following section of init through trajectory, relay and upgrade methods

        # Generate init train_X using given init method and probe the functions
        self.mission.train_X, self.mission.train_Y = self.generate_init_data()

        # Convert init train data to display data
        self.mission.display_X, self.mission.display_Y = self.generate_display_data()

        # Log init display data
        data_dict = self.generate_log_data(init = True) 
        self.mission.write_to_logfile(data = data_dict)
        

    
    def generate_init_data(self):
        
        if self.init_method == 'grid':
            init_input = init_grid(self.mission.envelope, self.num_init_design)
        else:
            pass

        # Initial Input Scaling
        if self.input_scaling:
            init_input = normalize(init_input, self.mission.envelope)

        # Initial Data Probing
        init_output = self.probe(input_data = init_input, init = True)

        return init_input, init_output
    
    def generate_display_data(self):
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

            
        if self.display_always_max:
            # Minimization (descend) objectives were inverted during optimization
            # Values of these objectives are now inverted back for displaying
            descend_indices = [idx for idx, value in enumerate(self.mission.maneuvers) if value == 'descend']
            display_output[:, descend_indices] *= -1
        else:
            pass

        return display_input, display_output
    
    def generate_log_data(self, init: bool = False):
        if init:
            display_X_subset = self.mission.display_X[-self.num_init_design:]
            display_Y_subset = self.mission.display_Y[-self.num_init_design:]
        else:
            display_X_subset = self.mission.display_X[-1]
            display_Y_subset = self.mission.display_Y[-1]

        if display_X_subset.dim() == 1 or display_Y_subset.dim() == 1:
            display_X_subset = display_X_subset.unsqueeze(0)
            display_Y_subset = display_Y_subset.unsqueeze(0)

        trajectory_dict = {f'param_{i+1}': display_X_subset[:, i].tolist() for i in range(display_X_subset.shape[1])}
        observation_dict = {f'objective_{i+1}': display_Y_subset[:, i].tolist() for i in range(display_Y_subset.shape[1])}
        data_dict = {**trajectory_dict, **observation_dict}

        return data_dict

    @abstractmethod
    def _upgrade(self, *args, **kwargs):
        """Update model with specific requirements"""
        pass

    def upgrade(self, *args, **kwargs):
        self._upgrade()
    
    @abstractmethod
    def _trajectory(self, *args, **kwargs):
        """
        Translate model-specific parameter recommendation to compatible format (torch.tensor)
        """
        pass

    def trajectory(self, *args, **kwargs):
        return self._trajectory(*args, **kwargs)

    def relay(self, trajectory, observation, *args, **kwargs):

        """
        Update the training data and display data with the new trajectory and observation. 
        Retain standardization and scaling for training data, but revert to original scale for display data.
        Log display data to mission logfile

        Args:
            trajectory (torch.Tensor): The new trajectory.
            observation (torch.Tensor): The observation of the new trajectory.
        """

        # Relay train data
        self.mission.train_X = torch.cat((self.mission.train_X, trajectory))
        self.mission.train_Y = torch.cat((self.mission.train_Y, observation))

        # Relay display data 
        self.mission.display_X, self.mission.display_Y = self.generate_display_data()

        # Log data
        ## Case where only one trajectory point and one observation point is observed (q=1)
        data_dict = self.generate_log_data(init = False)
        self.mission.write_to_logfile(data = data_dict)
        
    
    def probe(self, input_data: torch.tensor, init: bool, *args, **kwargs):

        for f in range(len(self.mission.funcs)):
            
            # Convert input data if scaling enabled
            if self.input_scaling:
                if init:
                    input_data = unnormalize(input_data, self.mission.envelope)
            
            output = self.mission.funcs[f](input_data)

            # Ensure Maximization problem
            if self.mission.maneuvers[f] == 'descend':
                output = -output
            else:
                pass
                
            # Ensure > 1D output
            if output.dim() < 2:
                output = output.unsqueeze(-1)
            
            # Concatenate multiple outputs
            if f == 0:
                output_all = output
            else:
                output_all = torch.cat((output_all, output), dim=1)

        # Calculate initial data mean and std
        if init:
            self.init_train_Y_mean = output_all.mean(dim=0)
            self.init_train_Y_std = output_all.std(dim=0)

        # Perform Data Standardization based on initial data mean and std
        if self.data_standardization:
            output_all = standardize(output_all, 
                                    mean = self.init_train_Y_mean, 
                                    std = self.init_train_Y_std
                                    )
                
            
        
        return output_all

                
            
       


