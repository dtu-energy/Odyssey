# Mission Class

from abc import ABC
from typing import Union

import torch
torch.set_default_dtype(torch.float64)

import numpy as np
import pandas as pd
import os
import datetime

class Mission(ABC):
    """
    Say something about the mission, boys
    """

    def __init__(self, 
                 name: str,
                 funcs: list, 
                 maneuvers: list, 
                 envelope: Union[list, np.ndarray, torch.Tensor]
        ):  

        # TODO If mission with same name already exists, do something

        for index, maneuver in enumerate(maneuvers):
            if maneuver not in ['ascend', 'descend']:
                raise ValueError(f"Maneuver '{maneuver}' at index {index} is invalid. Maneuvers must be either 'ascend' or 'descend'")

        self.funcs = funcs
        self.maneuvers = maneuvers
        self.envelope = torch.tensor(envelope)

        self.param_dims = self.envelope.shape[0]
        self.output_dims = len(self.maneuvers)
        
        # Setup Logfile
        self.name = name

        log_dir = "missionlogs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_time = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
        self.logfile = os.path.join(log_dir, f"{self.name}-{current_time}.csv")

        # Setup Logfile Columns
        param_list = [f"param_{i}" for i in range(1, self.param_dims + 1)]
        objective_list = [f"objective_{i}" for i in range(1, self.output_dims + 1)]

        self.columns = ['creation_timestamp'] + param_list + objective_list

        # Write Columns to Logfile
        log_df = pd.DataFrame(columns=self.columns)
        log_df.to_csv(self.logfile, index=False)

    def read_logfile(self) -> pd.DataFrame:
        return pd.read_csv(self.logfile)

    def write_to_logfile(self, data: dict):

        data['creation_timestamp'] = datetime.datetime.now().strftime("%d-%m-%y %H:%M:%S")

        assert set(data.keys()) == set(self.columns), "Data dict keys do not match logfile columns"

        # Open logfile
        log_df = self.read_logfile()

        # Append data to logfile
        append_df = pd.DataFrame(data, index = [0])

        if set(append_df.columns) == set(log_df.columns):
            if len(log_df) == 0:
                log_df = append_df
            else:
                log_df = pd.concat([log_df, append_df], ignore_index = True).reset_index(drop=True)

        # Write to logfile
        log_df.to_csv(self.logfile, index=False)
        print(f"Succesfully appended {data} to {self.logfile}")
        

    