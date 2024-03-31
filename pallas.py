import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
from functools import partial


def matmul_kernel(x_ref, y_ref, acc_ref, z_ref, *, n_steps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += jnp.dot(x_ref[...], y_ref[...], preferred_element_type=jnp.float32)

    @pl.when(pl.program_id(2) == n_steps - 1)
    def _():
        z_ref[...] = acc_ref[...].astype(z_ref.dtype)


@partial(jax.jit, static_argnames=("bm", "bk", "bn"))
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
    bn: int = 128,
):
    m, k = x.shape
    _, n = y.shape
    grid = (m // bm, n // bn, k // bk)
    print(m, k, n)
    print(grid)
    acc = jnp.zeros((m, n), dtype=jnp.float32)
    return pl.pallas_call(
        partial(matmul_kernel, n_steps=k // bk),
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        in_specs=[
            pl.BlockSpec(lambda i, j, k: (i, k), (bm, bk)),
            pl.BlockSpec(lambda i, j, k: (k, j), (bk, bn)),
            pl.BlockSpec(lambda i, j, k: (i, j), (bm, bn)),
        ],
        out_specs=pl.BlockSpec(lambda i, j, k: (i, j), (bm, bn)),
        grid=grid,
    )(x, y, acc)


if __name__ == "__main__":
    m = k = n = 128 * 20
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    # use float16 instead of bfloat16 since the GPU is a GTX 3090
    x = jax.random.normal(k1, (m, k), dtype=jnp.float16)
    y = jax.random.normal(k2, (k, n), dtype=jnp.float16)
    # x = jax.random.normal(k1, (m, k), dtype=np.float32)
    # y = jax.random.normal(k2, (k, n), dtype=jnp.float32)
    # x = jnp.ones((m, k), dtype=jnp.bfloat16)
    # y = jnp.ones((k, n), dtype=jnp.bfloat16)
    result = matmul(x, y)
    expected = jnp.dot(x, y)
    print(result)
    print(expected)
    print(result.dtype, expected.dtype)
    print(
        "All close?",
        jnp.allclose(expected, result, atol=1e-2, rtol=0),
    )
    print("Success!")
