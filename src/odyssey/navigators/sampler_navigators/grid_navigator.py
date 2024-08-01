from odyssey.navigators import Navigator
import torch

class Grid_Navigator(Navigator):

    """
    Grid_Navigator is a subclass of the Navigator class that navigates the search space using a grid-based approach. 
    It does not require initial data.

    Attributes:
        requires_init_data (bool): A flag, set to False, indicating that this navigator does not require initial data.
        iter_value (int): The current iteration value.
        x (torch.Tensor): The tensor representing the grid.

    !!! warning
        As of yet, the Grid_Navigator does not function as a standalone navigator. It is only used for initial sampling.
    """

    requires_init_data = False

    def __init__(self,
                 subdivisions: int,
                 *args,
                 **kwargs):
        
        """
        Initializes a Grid_Navigator object. 

        Args:
            subdivisions (int): The number of equidistant subdivisions of the parameter space for the grid.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        super().__init__(*args, **kwargs)

        self.iter_value = 0
        
        tensor_dims = [
            (torch.randperm(subdivisions) * (upper - lower)) / ((subdivisions - 1) + lower) 
            for lower, upper in self.mission.envelope
        ]
        self.x = torch.stack(tensor_dims, dim=1)

    def _upgrade(self):

        """
        Simply increments the iteration value.
        """

        self.iter_value += 1

    def _get_next_trial(self) -> torch.Tensor:

        """
        Selects the next candidate from the pre-generated grid.

        Returns:
            torch.Tensor: The next candidate from the grid.
        """

        try:
            candidate = self.x[[self.iter_value]]
        except IndexError as e:
            raise IndexError("grid sequence exhausted")

        return candidate