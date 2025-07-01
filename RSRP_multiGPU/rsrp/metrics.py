import jax
import jax.numpy as jnp
from jax.nn import one_hot

import optax
from flax.core import FrozenDict

"""
Logits should have shape (sub_pop_size, batch_size, N)
Label_batch shoud have shape (sub_pop_size, batch_size)
Fitness should have shape (sub_pop_size, )
"""


def accuracy(logits, label_batch, mean_axis=-1):
    hit = (jnp.argmax(logits, -1) == label_batch)
    return jnp.mean(hit, axis=mean_axis)


def recall(logits, label_batch, mean_axis=(-1,)):
    pred_ind = jnp.argmax(logits, axis=-1)
    pred_onehot = one_hot(pred_ind, num_classes=logits.shape[-1])
    label_onehot = one_hot(label_batch, num_classes=logits.shape[-1])

    sum_axis = tuple(i - 1 for i in mean_axis)  # One hot introduce additional 1 dim, therefore need subtract 1 for axis
    true_positives = jnp.sum(pred_onehot * label_onehot, axis=sum_axis)
    actual_positives = jnp.sum(label_onehot, axis=sum_axis)
    recall = true_positives / actual_positives
    avg_recall = jnp.nanmean(recall, axis=-1)  # ignoring classes not in label
    return avg_recall


def softrecall(logits, label_batch, mean_axis=(-1,)):
    """
    A smoothed version of recall.
    Use macro averaging through classes for balancing.
    Score calculate if the logit is close enough to label. With a hit to label (logit[label] is largest), score = 1.
    """

    label_onehot = one_hot(label_batch, num_classes=logits.shape[-1])
    true_logits = jnp.sum(logits * label_onehot, axis=-1)
    greater_logits = logits >= jnp.expand_dims(true_logits, -1)
    count_greater_logits = jnp.sum(greater_logits, axis=-1)
    
    # Smoothing the accuracy calculation. Larger a means a slower decay of score
    a = 5
    score = a / (a + count_greater_logits)

    sum_axis = tuple(i - 1 for i in mean_axis)  # One hot introduce additional 1 dim, therefore need subtract 1 for axis
    true_positives = jnp.sum(jnp.expand_dims(score, -1) * label_onehot, axis=sum_axis)
    actual_positives = jnp.sum(label_onehot, axis=sum_axis)
    recall = true_positives / actual_positives
    avg_recall = jnp.nanmean(recall, axis=-1)  # ignoring classes not in label
    return avg_recall

def softacc(logits, label_batch, mean_axis=(-1,)):
    """
    A smoothed version of recall.
    Use macro averaging through classes for balancing.
    Score calculate if the logit is close enough to label. With a hit to label (logit[label] is largest), score = 1.
    """

    label_onehot = one_hot(label_batch, num_classes=logits.shape[-1])
    true_logits = jnp.sum(logits * label_onehot, axis=-1)
    greater_logits = logits >= jnp.expand_dims(true_logits, -1)
    count_greater_logits = jnp.sum(greater_logits, axis=-1)

    # Smoothing the accuracy calculation. Larger a means a slower decay of score
    a = 5
    score = a / (a + count_greater_logits)

    return jnp.mean(score, axis=-1)

def perplexity(logits, labels):
    return jnp.exp(cross_entrophy(logits, labels))


def cross_entrophy(logits, labels, mean_axis=-1):
    ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return -jnp.mean(ce_loss, axis=mean_axis)


def param_square_error(params: FrozenDict) -> dict:
    """
    Measure parameter squared error across different devices.
    """

    return _param_square_error(params)


@jax.jit
def _param_square_error(params: FrozenDict):
    def _body(_params):
        _mean = jnp.mean(_params, axis=0)
        err = jnp.mean((_params - _mean) ** 2)
        return err

    tree_err = jax.tree_map(_body, params)
    return tree_err.unfreeze()


def avg_entrophy(params: FrozenDict) -> dict:
    """
    Measure mean information entrphy of parameters
    """

    return _avg_entrophy(params)


@jax.jit
def _avg_entrophy(params: FrozenDict):
    def _body(_param):
        return -jnp.mean(_param * jnp.log2(_param) + (1 - _param) * jnp.log2(1 - _param))

    return jax.tree_map(_body, params).unfreeze()