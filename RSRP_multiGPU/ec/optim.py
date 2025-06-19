import optax
import jax
import jax.numpy as jnp

from typing import Any, Callable, Optional, Union

NO_PARAMS_MSG = (
    'You are using a transformation that requires the current value of '
    'parameters, but you are not passing `params` when calling `update`.')


def add_entrophy_decayed_weights(
        weight_decay: Union[float, jax.Array] = 0.0,
        mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None
) -> optax.GradientTransformation:
    """Add parameter scaled by `weight_decay`.
    Adopted from Optax.add_decayed_weights. Use entrophy edcay in EC.

    Args:
        weight_decay: A scalar weight decay rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(NO_PARAMS_MSG)

        def _body(_updates, _params):
            # Gradient of Information Entrophy
            decay = weight_decay * (jnp.log(_params) + jnp.log(1 - _params))
            return _updates - decay

        updates = jax.tree_map(_body, updates, params)

        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return optax.masked(
            optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)


def add_exponential_decayed_weights(
        weight_decay: Union[float, jax.Array] = 0.0,
        mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None
) -> optax.GradientTransformation:
    """Add parameter scaled by `weight_decay`.
    Adopted from Optax.add_decayed_weights. Use entrophy edcay in EC.

    Args:
        weight_decay: A scalar weight decay rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
        or a Callable that returns such a pytree given the params/updates.
        The leaves should be booleans, `True` for leaves/subtrees you want to
        apply the transformation to, and `False` for those you want to skip.

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(NO_PARAMS_MSG)

        def _body(_updates, _params):
            # Decaying to center value 0.5
            decay = weight_decay * (_params - 0.5)
            return _updates - decay

        updates = jax.tree_map(_body, updates, params)

        return updates, state

    # If mask is not `None`, apply mask to the gradient transformation.
    # E.g. it is common to skip weight decay on bias units and batch stats.
    if mask is not None:
        return optax.masked(
            optax.GradientTransformation(init_fn, update_fn), mask)
    return optax.GradientTransformation(init_fn, update_fn)

def layer_lr_by_entropy(
) -> optax.GradientTransformation:
    """
    Args:

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        
        def _body(_updates, _params):
            entropy = -jnp.mean(_params * jnp.log2(_params) + (1 - _params) * jnp.log2(1 - _params))
            return _updates * entropy

        updates = jax.tree_map(_body, updates, params)
        
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)

def layer_lr_by_norm(
) -> optax.GradientTransformation:
    """
    Args:

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        
        def _body(_updates, _params):
            return _updates / (jnp.linalg.norm(_updates)/_updates.size)

        updates = jax.tree_map(_body, updates, params)
        
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)

def layer_lr_by_sqrtnorm(
) -> optax.GradientTransformation:
    """
    Args:

    Returns:
        A `GradientTransformation` object.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params):
        
        def _body(_updates, _params):
            return _updates / jnp.sqrt(jnp.linalg.norm(_updates)/_updates.size)

        updates = jax.tree_map(_body, updates, params)
        
        return updates, state
    
    return optax.GradientTransformation(init_fn, update_fn)