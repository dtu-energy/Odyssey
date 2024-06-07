In the following a Single-Input-Single-Output (SISO) function is optimized.

The function that we will be optimizing is:


$$ f(x) = -\left(\sin{\left(x\right)} + \sin{\left(\frac{10}{3} \cdot x\right)}\right) $$

which is a maximization function where the parameter $x$ has a range of 0 to 10, and the optimum point is at $x^* = 5$ and $f(x^*) = 2$.

### 1. Import the necessary packages
```python
import torch
from odyssey.mission import Mission # Mission
from odyssey.objective import Objective # Objective
from odyssey.navigators import SingleGP_Navigator # Navigator
from odyssey.navigators.sampler_navigators import Sobol_Navigator # Initial Sampler
from odyssey.navigators import UpperConfidenceBound # Acquisition Function
```

### 2. Define the function and the objective
The function is defined, and a `noise_level` parameter is added to implement some measurement noise.
```python
def real_func(x: torch.Tensor, noise_level = 0):
    sub_result = [-(torch.sin(x_i) + torch.sin((10.0 / 3.0) * x_i)) + (-1 + torch.rand(1)[0] * 2) * noise_level for x_i in x]
    return torch.Tensor(sub_result)
```

The `real_func` function is wrapped by the `Objective` class to define the objective.
```python
objective = Objective(real_func)
```

### 3. Define the Mission

In this case, we have a Mission with one input dimension (`param_dims`) and one output dimension (`output_dims`). The `param_space` for the single input variable is 0 to 10, and as this is a maximization function, the goal is to ascend.

```python
param_dims = 1
output_dims = 1

param_space = [(0.0, 10.0)]
goals = ['ascend']

mission = Mission(
    name = 'siso_test',
    funcs = [objective],
    maneuvers = goals,
    envelope = param_space
)
```

### 4. Define the navigator
We want to initialize the model with 4 initial points, defined by the `num_init_design` variable, and the `SingleGP_Navigator` well suited as the navigator, as this is a single objective problem. Additionally, the `UpperConfidenceBound` (UCB) acquisition function is used with the `beta` parameter set to $0.2$.
```python
num_init_design = 4

navigator = SingleGP_Navigator(
    mission = mission,
    num_init_design = num_init_design,
    input_scaling = False,
    data_standardization = False,
    init_method = Sobol_Navigator(mission = mission),
    acq_function_type = UpperConfidenceBound,
    acq_function_params = {'beta': 0.2}
)
```

### 5. Run the optimization loop
The `num_iter` variable defines the number of iterations to run. Some additional packages are imported here to filter out any warnings that might arise during the loop. The intermediate trajectories and observations are printed while the loop runs.

```python 
num_iter = 10

from warnings import catch_warnings
from warnings import simplefilter

while len(mission.train_X) - num_init_design < num_iter:

    with catch_warnings() as w:
        simplefilter('ignore')
        
        trajectory = navigator.trajectory()
        observation = navigator.probe(trajectory, init = False)
        print(len(mission.train_X) - num_init_design, trajectory, observation)
        navigator.relay(trajectory, observation)
        navigator.upgrade()
```

<!-- TODO: Add some print statements, python outputs, plots and uncertainty plots -->
