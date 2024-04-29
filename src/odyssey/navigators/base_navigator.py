# Base Navigator Class

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
                 data_standardization: bool
        ):

        # FIXME Input scaling not working correctly.

        self.mission = mission
        self.num_init_design = num_init_design
        self.init_method = init_method

        self.input_scaling = input_scaling
        self.data_standardization = data_standardization

        if self.input_scaling:
            self.traj_bounds = torch.tensor([[0.0,1.0]]).T # Unit Cube bounds
            # TODO Change to multiple dimensinoal unit cube bounds
        else:
            self.traj_bounds = self.mission.envelope.T

        # Generate Initial train_X using given init method
        self.mission.train_X, self.mission.train_Y = self.generate_init_data()
        

    
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

    @abstractmethod
    def _upgrade(self, *args, **kwargs):
        """Method that returns the model specific to the navigator """
        pass

    def upgrade(self, *args, **kwargs):
        self._upgrade()
    
    @abstractmethod
    def _trajectory(self, *args, **kwargs):
        pass

    def trajectory(self, *args, **kwargs):
        return self._trajectory(*args, **kwargs)
        
    
    @abstractmethod
    def _relay(self, *args, **kwargs):
        pass

    def relay(self, *args, **kwargs):
        return self._relay(*args, **kwargs)
    
    def probe(self, input_data: torch.tensor, init: bool, *args, **kwargs):

        for f in range(len(self.mission.funcs)):
            
            # Convert input data if scaling enabled
            if self.input_scaling:
                if init:
                    input_data = unnormalize(input_data, self.mission.envelope)
            
            output = self.mission.funcs[f](input_data)

            # Ensure Maximization problem
            if self.mission.maneuvers[f] == 'ascend':
                pass
            elif self.mission.maneuvers[f] == 'descend':
                output = -output

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

                
            
       


