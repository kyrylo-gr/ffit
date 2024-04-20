from typing import Literal, Union

from .protocol import BackendProtocol

SCIPY = "scipy"
JAX = "jax"

_POSSIBLE_BACKENDS = Literal["scipy", "jax"]


def get_scipy_backend() -> BackendProtocol:
    from .scipy_backend import ScipyBackend

    return ScipyBackend


def get_jax_backend() -> BackendProtocol:
    from .jax_backend import JAXBackend

    return JAXBackend


def get_backend(backend: Union[Literal["scipy", "jax"], str]) -> BackendProtocol:
    if backend == SCIPY:
        return get_scipy_backend()
    elif backend == JAX:
        return get_jax_backend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


# backends: Dict[Literal["scipy", "jax"], BackendProtocol] = {
#     "scipy": get_scipy_backend(),
#     "jax": get_jax_backend(),
# }


class Backends:
    @classmethod
    def get(cls, __key: Literal["scipy", "jax"]) -> BackendProtocol:
        return get_backend(__key)
