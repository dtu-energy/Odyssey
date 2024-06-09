The Odyssey library provides a framework for Bayesian optimization. The main components of this library are the Mission, Objective, Navigator and Acquisition Function classes and are explained in the following.

## Mission Class

The `Mission` class is the central component of the Odyssey library, representing the optimization problem at hand. It is initialized with several key parameters:

- `funcs`: This is a list of functions to be optimized. Each function should take a tensor as input and return a tensor as output. The tensor dimensions should match the problem's dimensions.
- `maneuvers`: This is a list of goals for each function. Each goal can be either 'ascend' (maximize the function) or 'descend' (minimize the function).
- `envelope`: This defines the parameter space for the optimization problem. It is a list of tuples, each representing the lower and upper bounds for a parameter.

The `Mission` class also maintains the training data (`train_X` and `train_Y`), which are updated as the optimization process progresses, along with the display data (`display_X` and `display_Y`), which are logged and displayed to the user in the `missionlogs` folder.

## Objective Class
The `Objective` class in the Odyssey library serves as a wrapper for a function, ensuring that the output is correctly formatted as a PyTorch tensor with the appropriate dimensions. This is particularly beneficial when the function is utilized as an objective function in optimization tasks, such as those managed by the `Navigator` and `Mission` classes. 

The `Objective` class is initialized with a single parameter:

- `func`: This is the function to be wrapped. It should take a tensor as input and return a value that can be converted to a tensor.

An instance of the Objective class using a function can now be created. When calling this instance, it takes any number of positional and keyword arguments, passes them to the stored function, and processes the output. The output is converted to a PyTorch tensor if it's not already one, and an extra dimension is added if the tensor has less than 2 dimensions. The processed tensor is then returned. This is in line with the requirements of the `Navigator` and `Mission` classes.

## Navigator Classes

Navigators in Odyssey are the optimization algorithms. They guide the search for the optimal solution in the parameter space defined by the `Mission`. 

### Base Navigator

The `Navigator` class is the foundation for all other navigator classes in the Odyssey library. It provides the basic structure and methods that are common to all navigators. Each navigator is initialized with a `Mission` and an acquisition function. 

The important parameters for initializing a navigator are:

- `num_init_design`: This parameter specifies the number of initial design points. These are the points in the parameter space where the function is probed before the optimization process starts. The initial design points provide the initial training data for the GP model(s).
- `init_method`: This parameter specifies the method for generating the initial design points. It can be an instance of a Sampling Navigator, such as the Sobol_Navigator or Grid_Navigator. The init_method generates a sequence of points in the parameter space, which are used as the initial design points.
- `input_scaling`: This parameter enables the normalization or scaling of input parameters to a range between 0 and 1. This can be beneficial for certain models, as it ensures all inputs have the same scale, preventing any one input from disproportionately influencing the model's predictions.
- `data_standardization`: This parameter allows for the standardization of output data, transforming it to have a mean of zero and a unit variance. This can be advantageous for some models, as it can help to stabilize the learning process and potentially improve the model's performance.

!!! info "Info"
    Currently, the `input_scaling` and `data_standardization` parameters are experiencing some issues. It is recommended to avoid using them until these issues are resolved.

The `Navigator` class provides several key methods that are inherited by all other navigator classes. These methods form the core of the optimization process.

- `trajectory`: This method generates a new point in the parameter space to probe next. The point is selected based on the current GP model(s) and the acquisition function. The goal is to find a point that is expected to improve the objective function(s) based on the current knowledge.
- `probe`: This method probes the objective function(s) at the point selected by the trajectory method. It evaluates the function(s) at the selected point and obtains the function value(s).
- `relay`: This method updates the training data with the new observation obtained by the probe method. It adds the selected point to `train_X` and the function value(s) to `train_Y`. This updates the GP model(s) with the new observation.
- `upgrade`: This method updates the GP model(s) based on the updated training data (standardized and normalized if required). It re-fits the model(s) to the new data, improving the approximation of the objective function(s) based on the new knowledge.

When initialized, the navigator takes the `init_method` and `num_init_data`, along with the `input_scaling` and `data_standardization` parameters and generates the initial data using a sequence of the `trajectory` and `probe` methods, and the generated data is logged. When actually optimizing, these methods are also called in sequence during each iteration of the optimization process. The `trajectory` method selects a point, the `probe` method obtains the function value(s) at that point, the `relay` method updates the training data with the new observation, and the `upgrade` method updates the model(s) with the new data. This process is repeated until the desired number of iterations is reached.

Finally, the navigators are split into three main groups, the Single Objective, Multi Objective and Sampling Navigators. 

### Single Objective Navigators
Single Objective Navigators are designed to optimize a single objective function. They use a single Gaussian Process (GP) model to approximate the objective function and guide the search for the optimal solution using an acquisition function. The `SingleGP_Navigator` is an example of a single objective navigator.

### Multi Objective Navigators

Multi Objective Navigators are designed for multi-objective optimization problems. They use multiple GP models, one for each objective, to approximate the objective functions and guide the search for the Pareto optimal solutions. The `qNParEGO_Navigator` and `Gryffin_Navigator` are examples of multi-objective navigators. 

### Sampling Navigators
Sampler Navigators generate a sequence of points in the parameter space using the same framework as the other navigators. The `Sobol_Navigator` and `Grid_Navigator` are examples of sampler navigators. 

Sampler Navigators can be used as initialization methods for other navigators. For example, the `SingleGP_Navigator` can use the `Sobol_Navigator` to generate the initial design points. This allows the `SingleGP_Navigator` to start the optimization process with a diverse set of points in the parameter space.

However, Sampler Navigators can also be used as standalone navigators. In this case they do not start with any inital points, but they generate a sequence of points in the parameter space, and the function is probed at each point. This can be useful for problems where the objective function is cheap to evaluate, or when a broad exploration of the parameter space is desired.

### DataLoader
The DataLoader, a unique variant of the Sampler Navigator, is designed to initialize your mission with data from its previous runs. It requires data files in a specific format. Provided that the number of parameters and objectives align, the DataLoader seamlessly integrates the data into your mission. 

## Acquisition Functions

Acquisition functions play a crucial role in Bayesian optimization. They guide the selection of the next point to probe in the parameter space, balancing the trade-off between exploration of uncharted areas and exploitation of known good regions. The goal is to select a point that is expected to yield the most valuable information for the optimization process.

Acquisition functions take as input the current state of the Gaussian Process (GP) model and return a utility value for each point in the parameter space. The point with the highest utility value is selected as the next point to probe. Different acquisition functions use different strategies to compute the utility value, and they can take different parameters to control their behavior.

In Odyssey, the acquisition functions are provided by the [BoTorch](https://botorch.org/) library. BoTorch is a flexible, modern library for Bayesian optimization that is built on [PyTorch](https://pytorch.org/). It provides a variety of acquisition functions, and these functions are well-tested and widely used in the field of Bayesian optimization.