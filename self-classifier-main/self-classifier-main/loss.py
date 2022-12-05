import jax
import jax.numpy as jnp
from jax.nn import softmax
from functools import partial

EPSILON = 1e-12


@partial(jax.jit, static_argnames=["axis"])
def norm(arr, *, axis):
    return arr / jnp.maximum(
        EPSILON, jnp.sum(jnp.abs(arr), axis=axis, keepdims=True)
    )


@jax.jit
def calc_log_normalized_y_cond_x(logits, t_row):
    p_y_cond_x = softmax(logits / t_row, axis=1)
    return jnp.log(norm(p_y_cond_x, axis=0) + EPSILON)


@jax.jit
def calc_normalized_x_cond_y(logits, t_col):
    p_x_cond_y = softmax(logits / t_col, axis=0)
    return norm(p_x_cond_y, axis=1)


@jax.jit
def loss(logits1, logits2, t_row, t_col):
    """
    Returns symmetric and asymmetric losses
    for grade very good experiment
    """
    N, C = logits1.shape
    log_NC = jnp.log(N / C)

    log_y_x1 = log_NC + calc_log_normalized_y_cond_x(logits1, t_row)
    log_y_x2 = log_NC + calc_log_normalized_y_cond_x(logits2, t_row)

    y_x1 = calc_normalized_x_cond_y(logits1, t_col)
    y_x2 = calc_normalized_x_cond_y(logits2, t_col)

    l1 = -jnp.sum(y_x2 * log_y_x1) / N
    l2 = -jnp.sum(y_x1 * log_y_x2) / N
    L = (l1 + l2) / 2
    return l1, l2, L


@jax.jit
def symmetric_loss(logits1, logits2, t_row, t_col):
    N, C = logits1.shape
    log_NC = jnp.log(N / C)

    log_y_x1 = log_NC + calc_log_normalized_y_cond_x(logits1, t_row)
    log_y_x2 = log_NC + calc_log_normalized_y_cond_x(logits2, t_row)

    y_x1 = calc_normalized_x_cond_y(logits1, t_col)
    y_x2 = calc_normalized_x_cond_y(logits2, t_col)

    l1 = -jnp.sum(y_x2 * log_y_x1, axis=1) / N
    l2 = -jnp.sum(y_x1 * log_y_x2, axis=1) / N
    L = (l1 + l2) / 2
    return jnp.mean(L)
