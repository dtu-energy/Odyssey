import numpy as np
import torch

def hartmann6(X):
    """Hartmann6 function (6-dimensional with 1 global minimum)."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 10 ** (-4) * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = 0
        for k in range(6):
            t += A[j, k] * ((X[k] - P[j, k]) ** 2)
        y -= alpha_j * np.exp(-t)
    
    return y


def hartmann6_torch(X):
    """Hartmann6 function (6-dimensional with 1 global minimum) using PyTorch."""
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 10 ** (-4) * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    
    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = torch.sum(A[j] * (X - P[j]) ** 2)
        y -= alpha_j * torch.exp(-t)
    
    return y

def hartmann6_torch_dict(input_dict):
    """Hartmann6 function (6-dimensional with 1 global minimum) using PyTorch with dictionary input."""
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ])
    P = 10 ** (-4) * torch.tensor([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ])
    
    # Convert the dictionary to a tensor
    X = torch.tensor([input_dict[f'x{i}'] for i in range(1, 7)])

    y = 0.0
    for j, alpha_j in enumerate(alpha):
        t = torch.sum(A[j] * (X - P[j]) ** 2)
        y -= alpha_j * torch.exp(-t)
    
    return {"hartmann6": y}

