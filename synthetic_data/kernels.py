import numpy as np

def rbf_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    # Broadcasting: (N, 1) - (1, M) -> (N, M)
    diff = (x1[:, None] - x2[None, :]) / tau
    return np.exp(-0.5 * diff**2)


def matern_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    nu = params.get("nu", 1.5)
    dist = np.abs(x1[:, None] - x2[None, :]) / tau
    
    if nu == 0.5:
        return np.exp(-dist)
    elif nu == 1.5:
        sqrt3_dist = np.sqrt(3) * dist
        return (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
    elif nu == 2.5:
        sqrt5_dist = np.sqrt(5) * dist
        return (1 + sqrt5_dist + 5 / 3 * dist**2) * np.exp(-sqrt5_dist)
    else:
        raise NotImplementedError(
            "Only nu=0.5, 1.5, 2.5 are implemented for Matern kernel"
        )


def periodic_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    period = params.get("period", 1.0)
    dist = np.abs(x1[:, None] - x2[None, :])
    return np.exp(-2 * (np.sin(np.pi * dist / period) / tau) ** 2)


KERNELS = {"rbf": rbf_kernel, "matern": matern_kernel, "periodic": periodic_kernel}