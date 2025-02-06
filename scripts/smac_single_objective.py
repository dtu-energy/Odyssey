import torch

from odyssey.mission import Mission # Mission
from odyssey.objective import Objective # Objective
# from odyssey.navigators import SingleGP_Navigator # Navigator
# from odyssey.navigators.sampler_navigators import Sobol_Navigator # Initial Sampler
# from odyssey.navigators import UpperConfidenceBound, ExpectedImprovement # Acquisition Function

from odyssey.navigators.smac_navigator import SMACNavigator

import logging

LOG = logging.getLogger(__name__)

def real_func(x: torch.Tensor, noise_level = 0):
    noise = (-1 + torch.rand(x.size()) * 2) * noise_level
    return -(torch.sin(x) + torch.sin((10.0 / 3.0) * x)) + noise


def main():
    import argparse

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=10,
        help="total number of iterations (initial + optimisation)",
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="number of initial points sampled",
    )

    parser.add_argument(
        "--log-level",
        choices=["info", "debug", "warning",],
        default="info",
        type=str,
    )

    args = parser.parse_args()
    num_init_design = args.n_init
    num_iter = args.iterations
    
    logging.basicConfig(level=args.log_level.upper())


    # set up mission
    test_X = torch.linspace(0, 10, 1000)
    test_Y = real_func(test_X, noise_level = 0)

    noise_level = 0.0
    objective = Objective(real_func, noise_level = noise_level)

    param_space = [(0.0, 10.0)]
    goals = ['ascend']

    params = [
        {
            "name": "x",
            "envelope": [0.0, 10.0],
            "type": "numerical"
        }
    ]

    objectives = [
        {
            "name": "f(x)",
            "maneuver": "ascend",
            "func": objective,
        }
    ]

    mission = Mission(
        name = 'siso_test',
        parameters=params,
        objectives=objectives,
        log_data=False,
    )


    # set up navigator
    # navigator = SingleGP_Navigator(
    #     mission = mission,
    #     n_init = num_init_design,
    #     # input_scaling = True,
    #     # data_standardization = True,
    #     init_method = "grid", #Sobol_Navigator(mission = mission, nsamples=num_init_design),
    #     # acq_function_type = "probability_of_improvement",
    #     acq_function_type = "upper_confidence_bound",
    #     # acq_function_params = {'beta': 0.5},
    #     # acq_function_type = ExpectedImprovement,
    #     # acq_function_params = {'best_f': 0.},
    # )

    navigator = SMACNavigator(
        mission=mission,
        n_init=num_init_design,
        n_trials=num_iter,
    )

    import pdb; pdb.set_trace()
    # BO loop
    for i in range(1, num_iter+1):
        params = navigator.get_next_trial()
        observation = navigator.probe(params, init = False)
        LOG.info("iter = {}; params = {}; observation = {}".format(
            i,
            ','.join([str(x.item()) for x in params.ravel()]), 
            ','.join([str(x.item()) for x in observation.ravel()]),
        ))
        navigator.relay(params, observation)
        if navigator.requires_init_data and i >= navigator.n_init: # move to relay or get_next_parameters so can work with any navigator
            navigator.upgrade()


    best_idx = mission.display_Y.argmax().item()
    best_input = mission.display_X[best_idx].item()
    best_output = mission.display_Y[best_idx].item()

    print(f'Best Input: {best_input}')
    print(f'Best Output: {best_output}')


if __name__ == "__main__":
    main()
