import jax
import jax.numpy as jnp
import flax.linen as nn

import math


from ec.ops import conn_dense#,conn_dense_ops
from ec.core import CONN_KERNEL



class Dense(nn.Module):
    """
        Dense Connection Module, with +1 / -1 kernel.
        Input should be Real numbers. For spiking networks with bool input, please use Bitset.
        Output vector is designed to keep input variance and have zero mean.
        TODO: Better efficiency computation kernel.
    """

    features : int
    dtype:  jnp.dtype = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):
        
        in_dim = x.shape[-1]
        kernel   = self.param(CONN_KERNEL, nn.initializers.zeros, (in_dim, self.features), jnp.bool_)
        kernel_inh = jnp.logical_not(kernel)

        # Normalization
        scale    = self.param("scale", nn.initializers.constant(1 / math.sqrt(in_dim), self.dtype), ()) 

        return scale*(conn_dense(kernel, x) - conn_dense(kernel_inh, x))

class MLP(nn.Module):
    
    features:   tuple[int, ...]
    dense_cls:  nn.Module = Dense
    dtype:      jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        
        for i in range(len(self.features)):
            x = self.dense_cls(features = self.features[i], dtype = self.dtype) (x)
            if i < len(self.features) - 1:
                x = nn.relu(x)*2

        return x



class Dense_01(nn.Module):
    """
        Dense Connection Module, with +1 / -1 kernel.
        Input should be Real numbers. For spiking networks with bool input, please use Bitset.
        Output vector is designed to keep input variance and have zero mean.
        TODO: Better efficiency computation kernel.
    """

    features : int
    dtype:  jnp.dtype = jnp.bfloat16
    
    @nn.compact
    def __call__(self, x):
        
        in_dim = x.shape[-1]
        kernel   = self.param(CONN_KERNEL, nn.initializers.zeros, (in_dim, self.features), jnp.bool_)
        # Normalization
        scale    = self.param("scale", nn.initializers.constant(1 / math.sqrt(in_dim), self.dtype), ()) 

        return scale*conn_dense(kernel, x)
    
class Dense_EI(nn.Module):
    """
        the neurons are devided into exi/inh types.
    """

    features : int
    dtype:  jnp.dtype = jnp.bfloat16
    p: float = 1
    
    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        num_excitatory = round(in_dim * self.p)
        x_exc, x_inh = jnp.split(x, [num_excitatory], axis=-1)
        x = jnp.concatenate([x_exc, - self.p/(1-self.p)*x_inh], axis=-1)

        kernel   = self.param(CONN_KERNEL, nn.initializers.zeros, (in_dim, self.features), jnp.bool_)
        # Normalization
        scale    = self.param("scale", nn.initializers.constant(1 / math.sqrt(in_dim), self.dtype), ()) 

        return scale*conn_dense(kernel, x)

class Dense_input(nn.Module):
    """
        the neurons are devided into exi/inh types.
    """

    features : int
    dtype:  jnp.dtype = jnp.bfloat16
    p: float = 1
    
    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]*2
        x = jnp.concatenate([x, - x], axis=-1)

        kernel   = self.param(CONN_KERNEL, nn.initializers.zeros, (in_dim, self.features), jnp.bool_)
        scale    = self.param("scale", nn.initializers.constant(1 / math.sqrt(in_dim), self.dtype), ()) 

        return scale*conn_dense(kernel, x)

        
class MLP_EI(nn.Module):
    
    features:   tuple[int, ...]
    dense_cls0:  nn.Module = Dense_01
    dense_cls1:  nn.Module = Dense_EI
    dtype:      jnp.dtype = jnp.bfloat16
    p:          float = 1

    @nn.compact
    def __call__(self, x):
        
        for i in range(len(self.features)):
            if i == 0:
                x = self.dense_cls0(features = self.features[i], dtype = self.dtype) (x)
            else:
                x = self.dense_cls1(features = self.features[i], dtype = self.dtype, p = self.p) (x)
            if i < len(self.features) - 1:
                x = nn.relu(x)*2

        return x

class MLP_EI_input(nn.Module):
    
    features:   tuple[int, ...]
    dense_cls0:  nn.Module = Dense_input
    dense_cls1:  nn.Module = Dense_EI
    dtype:      jnp.dtype = jnp.bfloat16
    p:          float = 1

    @nn.compact
    def __call__(self, x):
        
        for i in range(len(self.features)):
            if i == 0:
                x = self.dense_cls0(features = self.features[i], dtype = self.dtype) (x)
            else:
                x = self.dense_cls1(features = self.features[i], dtype = self.dtype, p = self.p) (x)
            if i < len(self.features) - 1:
                x = nn.relu(x)*2

        return x
        
class MLP_01(nn.Module):
    
    features:   tuple[int, ...]
    dense_cls0:  nn.Module = Dense_input
    dense_cls1:  nn.Module = Dense_01
    dtype:      jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        
        for i in range(len(self.features)):
            if i == 0:
                x = self.dense_cls0(features = self.features[i], dtype = self.dtype) (x)
            else:
                x = self.dense_cls1(features = self.features[i], dtype = self.dtype) (x)
            if i < len(self.features) - 1:
                x = nn.relu(x)*2

        return x