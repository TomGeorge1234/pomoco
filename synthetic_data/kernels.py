import jax.numpy as jnp


def rbf_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    diff = (x1[:, None] - x2[None, :]) / tau
    return jnp.exp(-0.5 * diff**2)


def matern_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    nu = params.get("nu", 1.5)
    dist = jnp.abs(x1[:, None] - x2[None, :]) / tau
    if nu == 0.5:
        return jnp.exp(-dist)
    elif nu == 1.5:
        sqrt3_dist = jnp.sqrt(3) * dist
        return (1 + sqrt3_dist) * jnp.exp(-sqrt3_dist)
    elif nu == 2.5:
        sqrt5_dist = jnp.sqrt(5) * dist
        return (1 + sqrt5_dist + 5 / 3 * dist**2) * jnp.exp(-sqrt5_dist)
    else:
        raise NotImplementedError(
            "Only nu=0.5, 1.5, 2.5 are implemented for Matern kernel"
        )


def periodic_kernel(x1, x2, params):
    tau = params.get("tau", 1.0)
    period = params.get("period", 1.0)
    dist = jnp.abs(x1[:, None] - x2[None, :])
    return jnp.exp(-2 * (jnp.sin(jnp.pi * dist / period) / tau) ** 2)


KERNELS = {"rbf": rbf_kernel, "matern": matern_kernel, "periodic": periodic_kernel}