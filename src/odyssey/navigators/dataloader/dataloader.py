from odyssey.navigators import Navigator
import torch
import pandas as pd

class DataLoader(Navigator):

    requires_init_data = False

    def __init__(self,
                 datafiles: list,
                 *args,
                 **kwargs):
        
        super().__init__(*args, **kwargs)

        self.param_columns = [f'param_{i+1}' for i in range(self.mission.param_dims)]
        self.objective_columns = [f'objective_{i+1}' for i in range(self.mission.output_dims)]
        self.mission_columns = self.param_columns + self.objective_columns
        self.data_df = pd.DataFrame(columns = self.mission_columns)

        for datafile in datafiles:
            datafile_df = pd.read_csv(datafile)
            assert set(datafile_df.columns) == set(self.mission_columns + ['creation_timestamp']), f'Columns in datafile {datafile} do not match mission data columns.'

            if len(self.data_df) == 0:
                self.data_df = datafile_df[self.mission_columns]
            else:
                self.data_df = pd.concat([self.data_df, datafile_df[self.mission_columns]], ignore_index = True).reset_index(drop = True)
            
            
    def _upgrade(self):
        pass

    def _trajectory(self):
        pass

    def load_input_data(self):
        loaded_input = torch.from_numpy(self.data_df[self.param_columns].values)
    
        #data_dict = self.generate_log_data(init = True) 
        #self.mission.write_to_logfile(data = data_dict)

        return loaded_input
    
    def load_output_data(self):
        loaded_output = torch.from_numpy(self.data_df[self.objective_columns].values)
        return loaded_output