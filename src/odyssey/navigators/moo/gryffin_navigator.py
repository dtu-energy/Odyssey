from typing import Type
import torch
torch.set_default_dtype(torch.float64)

from gryffin import Gryffin
from odyssey.navigators import Navigator

class Gryffin_Navigator(Navigator):

    def __init__(self,
                 tolerances,
                 absolute,
                 param_types,
                 general_config,
                 model_config,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.general_config = general_config
        self.model_config = model_config
        self.tolerances = tolerances
        self.absolute = absolute
        self.param_types = param_types

        self.create_config()
        self.create_observations()

        self.upgrade()


    def create_config(self):
        config = {
            'general': self.general_config,
            'model': self.model_config,
            'parameters': [],
            'objectives': []
            }
        
        # Parameters
        for i in range(len(self.mission.envelope)):
            config['parameters'].append({
                'name': f'param_{i}',
                'type': self.param_types[i],
                'low': self.mission.envelope[i][0].item(),
                'high': self.mission.envelope[i][1].item(),
                'size': 1
                })
        
        # Objectives
        for i in range(len(self.mission.funcs)):
            config['objectives'].append({
                'name': f'obj_{i}',
                'goal': 'max', # Always maximization
                'tolerance': self.tolerances[i],
                'absolute': self.absolute[i]
                })
            
        self.config = config
            
    def create_observations(self):
        
        # Observations
        observations = []
        for x, y in zip(self.mission.train_X, self.mission.train_Y):
            observation = {}
            for i, param in enumerate(self.config['parameters']):
                observation[param['name']] = x[i].item()
            for i, obj in enumerate(self.config['objectives']):
                observation[obj['name']] = y[i].item()
            observations.append(observation)
        
        self.observations = observations


    def _upgrade(self):
        self.model = Gryffin(config_dict = self.config, silent = True)

    def _trajectory(self):
        
        candidate = self.model.recommend(observations = self.observations)
        candidate = candidate[0] # Only use the first recommendation

        for i in range(len(candidate)):
            keys = list(candidate.keys())
            if i == 0:
                final_candidate = torch.tensor([candidate[keys[i]]])
            else:
                final_candidate = torch.cat((final_candidate, torch.tensor([candidate[keys[i]]])))

        final_candidate = final_candidate.unsqueeze(0)

        return final_candidate

    def _relay(self, trajectory: torch.Tensor, observation: torch.Tensor):
        # Update train X and Y
        self.mission.train_X = torch.cat((self.mission.train_X, trajectory))
        self.mission.train_Y = torch.cat((self.mission.train_Y, observation))
        
        # Update Gryffin-specific observations
        self.create_observations()