from odyssey.navigators import Navigator
import torch
import pandas as pd

import logging
LOG = logging.getLogger(__name__)

class DataLoader(Navigator):

    """DataLoader is a subclass of the Navigator class that loads data from log files as initialization. It does not require initial data.
    
    The files being loaded assume that the data is stored in a .csv format with the columns matching with the columns of the attached mission. 
    The names of these columns must be in the format `param_1`, `param_2`, ..., `param_n`, `objective_1`, `objective_2`, ..., `objective_m`, where `n` is the number of parameters and `m` is the number of objectives in the mission.

    Attributes:
        requires_init_data: A flag, set to False, indicating that this navigator does not require initial data.
        param_columns: A list of column names for parameters in the data.
        objective_columns: A list of column names for objectives in the data.
        mission_columns: A list of all column names in the data.
        data_df: A DataFrame holding the data loaded from files.
        iter_value: An integer iterator for the data_df.

    Examples:
        Assuming that you have some data, you can load them using the DataLoader as follows:
        >>> from odyssey.navigators import DataLoader
        ...
        >>> datafiles = ['missionlogs/MISSION1_NAME.csv', 'missionlogs/MISSION2_NAME.csv']
        >>> dl = DataLoader(mission = mission, datafiles = datafiles)

    !!! warning
        As of yet, the DataLoader does not function as a standalone navigator. It can only be used for initial sampling.
    """

    requires_init_data = False

    def __init__(self,
                 datafiles: list,
                 *args,
                 **kwargs):
        
        """
        Initializes a DataLoader object.
        
        Args:
            datafiles: A list of paths to the data files.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Raises:
            AssertionError: If the number of initial design points is greater than the number of points in the combined datafiles.

        """

        
        super().__init__(*args, **kwargs)

        

        self.param_columns = [f'param_{i+1}' for i in range(self.mission.param_dims)]
        self.objective_columns = [f'objective_{i+1}' for i in range(self.mission.output_dims)]
        self.mission_columns = self.param_columns + self.objective_columns
        self.data_df = pd.DataFrame(columns = self.mission_columns)

        for datafile in datafiles:
            datafile_df = pd.read_csv(datafile)
    
            if not set(datafile_df.columns) == set(self.mission_columns + ['creation_timestamp']):
                LOG.warning(f'Columns in datafile {datafile} do not match mission data columns. Skipping this file.')
            else:
                if len(datafile_df) == 0: # If no data in the datafile
                    pass
                else:
                    if len(self.data_df) == 0:
                        self.data_df = datafile_df[self.mission_columns]
                    else:
                        self.data_df = pd.concat([self.data_df, datafile_df[self.mission_columns]], ignore_index = True).reset_index(drop = True)
            
        self.data_df = torch.Tensor(self.data_df.filter(like = 'param', axis = 1).to_numpy())

        self.iter_value = 0
            
    def _upgrade(self):

        """Simply increments the iteration value.
        """
        
        self.iter_value += 1

    def _trajectory(self) -> torch.Tensor:

        """
        Selects the next candidate from the pre-loaded datafile points.

        Returns:
            torch.Tensor: The next candidate from the datafiles.
        """

        candidate = self.data_df[[self.iter_value]]
        return candidate