import torch
import numpy as np

def apply_randomization(tensor, params, return_noise=False):
    if params == None:
        return tensor

    if params["distribution"] == "gaussian":
        mu, var = params["range"]
        noise = torch.randn_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.randn()
        noise_val = mu + var * noise
    elif params["distribution"] == "uniform":
        lower, upper = params["range"]
        noise = torch.rand_like(tensor) if isinstance(tensor, torch.Tensor) else np.random.rand()
        noise_val = lower + (upper - lower) * noise
    else:
        raise ValueError(f"Invalid randomization distribution: {params['distribution']}")

    if params["operation"] == "additive":
        result = tensor + noise_val
    elif params["operation"] == "scaling":
        result = tensor * noise_val
    else:
        raise ValueError(f"Invalid randomization operation: {params['operation']}")

    if return_noise:
        return result, noise
    else:
        return result