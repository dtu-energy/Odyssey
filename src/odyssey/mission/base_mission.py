# Mission Class

from typing import Union
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from abc import ABC, abstractmethod

class Mission(ABC):
    def __init__(self, 
                 funcs: Union[list, np.ndarray, torch.Tensor], 
                 maneuvers: Union[list, np.ndarray, torch.Tensor],  # TODO throw some error if maneuver not "ascend" or "descend"
                 envelope: Union[list, np.ndarray, torch.Tensor]
        ):      
        
        self.funcs = funcs
        self.maneuvers = maneuvers
        self.envelope = torch.tensor(envelope)        
    