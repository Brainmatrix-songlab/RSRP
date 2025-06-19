import math
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.nn.initializers import _compute_fans
from jax import core

from .linear import Dense
from ec.core import CONN_KERNEL

import numpy as np
import math

from typing import (
    Optional,
    Sequence,
    Tuple,
    Union,
)

def _conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

class ConvConn(nn.Module):
    '''Conv block with only bool connection as kernel'''

    features: int
    kernel_size: Tuple[int]
    strides: int = 1
    padding: str = 'VALID'
    precision: jnp.dtype = jnp.bfloat16 

    @nn.compact
    def __call__(self, inputs):
        
        kernel_size = self.kernel_size
        
        def maybe_broadcast(
            x: Optional[Union[int, Sequence[int]]]
        ) -> Tuple[int, ...]:
            if x is None:
                # backward compatibility with using None as sentinel for
                # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)
        strides = maybe_broadcast(self.strides)

        # 保证输入矩阵为[bs,ch,w,w]
        dimension_numbers = lax.ConvDimensionNumbers(lhs_spec=(0,3,1,2), rhs_spec=(3,2,0,1), out_spec=(0,3,1,2))

        in_features = jnp.shape(inputs)[-1]
        kernel_shape = kernel_size+(in_features,self.features,)
        kernel = self.param(CONN_KERNEL, nn.initializers.zeros, kernel_shape, jnp.bool_)

        lhs = inputs.astype(self.precision)
        kernel_inh = jnp.logical_not(kernel)
        rhs = kernel.astype(self.precision)-kernel_inh.astype(self.precision)
        y = jax.lax.conv_general_dilated(
            lhs = lhs,
            rhs = rhs,
            window_strides = strides,
            padding = self.padding,
            dimension_numbers = dimension_numbers,
        )

        # # Normalization
        # namedshape = core.as_named_shape(kernel_shape)
        # fan_in, _ = _compute_fans(namedshape, -2, -1, ())
        # R = 1 / math.sqrt(fan_in)

        return y
    

class CNN(nn.Module):
    '''A simple CNN model for EC.'''
    kernel_size: list
    cnn_channels: list
    stride: list
    mlp_features: list

    @nn.compact
    def __call__(self, x):

        x = x.transpose(0,2,3,1)

        for i in range(len(self.cnn_channels)):
            x = ConvConn(features=self.cnn_channels[i], strides=self.stride[i], kernel_size=self.kernel_size)(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))  # flatten

        for i in range(len(self.mlp_features)):
            x = Dense(features=self.mlp_features[i]) (x)
            if i < len(self.mlp_features) - 1:
                x = nn.relu(x)

        return x