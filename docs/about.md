# Overview on `FFit`.

## Overall motivation

The basic motivation can be divided into two parts:

- First, the current process of fitting with `scipy.optimize.curve_fit` is too lengthy and writing-code-consuming, while it should be straightforward.
- Second, with new technology like `JAX` for python; `ocl` or `wgpu` for `rust`, we can accelerate matrix multiplication, thus probably making the entire algorithm faster.

The first problem can be easily solved with a few hundred lines of code.
Therefore, this overview will primarily focus on the second problem: How can we speed up the algorithm?

Spoiler alert: there's a possible potential to speed up the runtime. However, due to my time constraints or impossibility, it hasn't been completed yet.

## Speed up the algorithm

Classically to fit some function you would use `scipy.optimize.curve_fit` or even `scipy.optimize.leastsq` method. The first is a wrap of the second and them both are using the
[Levenberg-Marquardt Algorithm](https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm) by default in the backend. It uses numpy library and [MINPACK](https://en.wikipedia.org/wiki/MINPACK) solver written on Fortran.

To make it faster one can use [JAX](https://github.com/google/jax). It uses CPU/TPU for matrix multiplication and has fast gradient calculating routine.

## Existed solutions

Several solutions already exist for this problem, with diverse implementations of the Levenberg–Marquardt algorithm, some of which utilize the Trust Region Method.

The first solution is the [JAXfit](https://github.com/Dipolar-Quantum-Gases/jaxfit?tab=readme-ov-file) library, developed by the University of Oxford. According to [their paper](https://arxiv.org/pdf/2208.12187), their library is ten times faster than the scipy method. However, I wasn't able to reproduce this result on my Mac with an M1 chip using different dataset lengths.

The second solution I found is [JAXOPT](https://github.com/google/jaxopt), developed by Google. In [their paper](https://arxiv.org/pdf/2105.15183), they demonstrate a decrease in runtime by a factor of four on large datasets using a powerful GPU. Despite this claim, I was unable to reproduce these results on my M1 Mac. Moreover, upon closer inspection, I noticed that the jaxopt library runs the function more frequently than the classical scipy, suggesting that their algorithm may not be as optimized.

It's worth noting that JAXOPT is currently transitioning to the new [OPTAX](https://github.com/google-deepmind/optax) library. However, as of now, the Levenberg–Marquardt algorithm is not available there.

With all this in mind, it motivated to explore ways to run functions faster than the traditional Python with Numpy. This is because, it still could be possible that without a powerful GPU and super-large datasets, Python and Numpy might already be quite optimized.

## Speed performance. Numba vs JAX

There are various ways to enhance the speed of Python code execution. One of the most popular methods is using Just-in-Time (JIT) compilers. When you run a typical Python code for the first time, the JIT compiler analyzes and compiles the code to make it faster for subsequent runs. Numba, a library that compiles your code to the CPU, is quite popular in this regard. However, JAX also has a JIT compiler and leverages the GPU for further calculations.

We are interested in comparing their performances. We test a classic trigonometric function with some matrix multiplications. For the dataset length, we chose 10,000 which is a quite large dataset for classical physics computations, but could highlight any GPU advantages over the CPU.

We are interested in comparing their performances. We test a classic trigonometric function with some matrix multiplications. For the dataset length, we chose 10,000 which is a quite large dataset for classical physics computations, but could highlight any GPU advantages over the CPU.

### Matrix sum

<details><summary>[See details]</summary>

```python
import jax.numpy as jnp
from numba import njit
import numpy as np
from jax import jit as jax_jit

@jax_jit
def jax_function(x):
    res = jnp.copy(x)
    for _ in range(100):
        res += x
    return res

@njit
def numba_function(x):
    res = np.copy(x)
    for _ in range(100):
        res += x
    return res

# Example data
x0 = np.random.rand(1_000_000).reshape(1000, 1000)
x_jax = jnp.array(x0)
x_numba = np.array(x0)

# Compile functions
jax_function(x_jax)
numba_function(x_numba)

# time the functions
%timeit jax_function(x_jax)
# 3.31 ms ± 290 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit numba_function(x_numba)
# 44.1 ms ± 3.85 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

</details>

### Martix product

<details><summary>[See details]</summary>

```python
import jax.numpy as jnp
from numba import njit
import numpy as np
from jax import jit as jax_jit

@jax_jit
def jax_function(x):
    res = jnp.copy(x)
    for _ in range(100):
        res += jnp.sin(x) @ jnp.cos(x)
    return res

@njit
def numba_function(x):
    res = np.copy(x)
    for _ in range(100):
        res += np.sin(x) @ np.cos(x)
    return res

# Example data
x0 = np.random.rand(1_000_000).reshape(1000, 1000)
x_jax = jnp.array(x0)
x_numba = np.array(x0)

# Compile functions
jax_function(x_jax)
numba_function(x_numba)

# time the functions
%timeit jax_function(x_jax)
# 19.3 ms ± 237 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

%timeit numba_function(x_numba)
# 7.28 s ± 1.44 s per loop (mean ± std. dev. of 7 runs, 1 loop each)

```

</details>

### Martix product on small arrays

<details><summary>[See details]</summary>

```python
import jax.numpy as jnp
from numba import njit
import numpy as np
from jax import jit as jax_jit

@jax_jit
def jax_function(x):
    res = jnp.copy(x)
    for _ in range(100):
        res += jnp.sin(x) @ jnp.cos(x)
    return res

@njit
def numba_function(x):
    res = np.copy(x)
    for _ in range(100):
        res += np.sin(x) @ np.cos(x)
    return res

# Example data
x0 = np.random.rand(100).reshape(10, 10)
x_jax = jnp.array(x0)
x_numba = np.array(x0)

# Compile functions
jax_function(x_jax)
numba_function(x_numba)

# time the functions
%timeit jax_function(x_jax)
# 12.5 µs ± 1.89 µs per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

%timeit numba_function(x_numba)
# 233 µs ± 3.06 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

</details>

### Need of JAX JIT compiling

<details><summary>[See details]</summary>

```python
import jax.numpy as jnp
from numba import njit
import numpy as np
from jax import jit as jax_jit

@jax.jit
def jax_function_nocc(x):
    res = jnp.copy(x)
    for _ in range(100):
        res += jnp.sin(x) @ jnp.cos(x)
    return res

@jax.jit
def jax_function_compile(x):
    res = jnp.copy(x)
    for _ in range(100):
        res += jnp.sin(x) @ jnp.cos(x)
    return res

# Compile functions

jax_function_nocc(x_jax)
jax_function_compile(x_jax)

# time the functions

%timeit jax_function_nocc(x_jax)
%timeit jax_function_compile(x_jax)

```

</details>
