import jax.numpy as jnp
import flax.linen as nn
import math

from ec.ops import conn_dense
from ec.core import CONN_KERNEL


class Dense(nn.Module):
    """
        Dense Connection Module, with +1 / -1 kernel.
        Input should be Real numbers. For spiking networks with bool input, please use Bitset.
        Output vector is designed to keep input variance and have zero mean.
    """

    features: int
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        in_dim = x.shape[-1]
        kernel = self.param(CONN_KERNEL, nn.initializers.zeros, (in_dim, self.features), jnp.bool_)
        kernel_inh = jnp.logical_not(kernel)

        # Normalization
        scale = self.param("scale", nn.initializers.constant(1 / math.sqrt(in_dim), self.dtype), ())
        return scale * (conn_dense(kernel, x) - conn_dense(kernel_inh, x))

class MlpBlock(nn.Module):
    mlp_dim: int
    @nn.compact
    def __call__(self, x):
        y = Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    """Mixer block layer."""
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)


class MlpMixer_ann(nn.Module):
    """Mixer architecture."""
    patch_size: int
    image_size: int
    num_classes: int
    num_blocks: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(self, inputs):
        patch_num = int(self.image_size / self.patch_size)
        inputs = inputs.transpose(0, 2, 3, 1)
        # cut into patches
        x = inputs.reshape(inputs.shape[0], patch_num, self.patch_size, patch_num, self.patch_size,
                                3).transpose(0, 1, 3, 2, 4, 5).reshape(inputs.shape[0], patch_num*patch_num,
                                                                       self.patch_size*self.patch_size*3)
        x = Dense(self.hidden_dim,name='token')(x)
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name='pre_head_layer_norm')(x)
        x = jnp.mean(x, axis=1)
        if self.num_classes:
            x = Dense(self.num_classes,name='head')(x)
        return x
