import torch
import pandas as pd

from odyssey.mission import Mission # Mission
from odyssey.objective import Objective # Objective
from odyssey.navigators.saasbo_navigator import BOTorch_SAASBO_Navigator # Navigator
from odyssey.navigators.sampler_navigators import Sobol_Navigator # Initial Sampler
# from odyssey.navigators import UpperConfidenceBound, ExpectedImprovement # Acquisition Function

from hartmann import hartmann6_torch

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
        default=30,
        help="total number of iterations (initial + optimisation)",
    )

    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
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


    all_X = []
    all_Y = []
    all_best_ys = []

    for i in range(5):

        # set up mission
        # test_X = torch.linspace(0, 10, 1000)
        # test_Y = real_func(test_X, noise_level = 0)

        objective = Objective(hartmann6_torch)

        params = [
            { 
                "name": f"x{i+1}",
                "envelope": [0.0, 1.0],
            } for i in range(6)
        ]

        objectives = [
            {
                "name": "hartmann6",
                "maneuver": "descend",
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
        navigator = BOTorch_SAASBO_Navigator(
            mission = mission,
            n_init = num_init_design,
            input_scaling = False, 
            data_standardization = False,
            # init_method = "grid",
            init_method="sobol",
            # acq_function = "probability_of_improvement",
            # acq_function = "upper_confidence_bound",
            acq_function = "expected_improvement",
            # acq_function_params = {'beta': 0.5},
            # acq_function = ExpectedImprovement,
            # acq_function_params = {'best_f': 0.},
        )

    
        # BO loop
        best_ys = []
        best_y = 1E9
        i = 1
        # import pdb; pdb.set_trace()
        while len(mission.train_X) < args.iterations:
            params = navigator.get_next_trial(q=1)
            observation = navigator.probe(params, init = False)
            if observation.item() < best_y:
                best_y = observation.item()
            LOG.info(f"best y = {best_y}")
            best_ys.append(best_y)
            LOG.info("iter = {}; params = {}; observations = {}".format(
                i,
                ','.join([str(x.item()) for x in params.ravel()]), 
                ','.join([str(x.item()) for x in observation.ravel()]),
            ))
            navigator.relay(params, observation)
            if i >= navigator.n_init: # move to relay or get_next_parameters so can work with any navigator
                navigator.upgrade()
            
            i += 1
        
        all_X.append(mission.display_X)
        all_Y.append(mission.display_Y)
        all_best_ys.append(-torch.tensor(best_ys))


    # stacked_Y = torch.stack(all_Y)
    # mean_y = torch.mean(stacked_Y, dim=0)
    # std_y = torch.std(stacked_Y, dim=0)
    # min_y = torch.min(stacked_Y, dim=0).values
    # max_y = torch.max(stacked_Y, dim=0).values

    stacked_Y = torch.stack(all_best_ys)
    mean_y = torch.mean(stacked_Y, dim=0)
    std_y = torch.std(stacked_Y, dim=0)
    min_y = torch.min(stacked_Y, dim=0).values
    max_y = torch.max(stacked_Y, dim=0).values

    # Create a DataFrame with entry numbers and average values
    df = pd.DataFrame({
        'iteration': range(1, num_iter + 1),  # Entry numbers start from 1
        'average_y': mean_y.flatten().tolist(),  # Convert tensor to a Python list
        "std_dev": std_y.flatten().tolist(),
        "min_y": min_y.flatten().tolist(),
        "max_y": max_y.flatten().tolist(),
    })

    # Write the DataFrame to a CSV file
    df.to_csv('optimisation_results.csv', index=False)

    # # final model
    # model = navigator.model
    # pred_mean = model.posterior(test_X).mean.detach().squeeze()
    # pred_std = torch.sqrt(model.posterior(test_X).variance).detach().squeeze()


    # best_idx = mission.display_Y.argmax().item()
    # best_input = mission.display_X[best_idx].item()
    # best_output = mission.display_Y[best_idx].item()

    # print(f'Best Input: {best_input}')
    # print(f'Best Output: {best_output}')


if __name__ == "__main__":
    main()
