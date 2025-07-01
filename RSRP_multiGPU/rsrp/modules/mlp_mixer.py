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
        TODO: Better efficiency computation kernel.
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


class LIF(nn.Module):
    """
        input: x(weighted spike from last layer),V
        params: V_threshold, tau
        output: x(spike),V_next
    """

    @nn.compact
    def __call__(self, x, V, V_thr=1.0, tau=2.0, V_reset=0.0):
        V_next = V + x - 1. / tau * (V - V_reset)
        x = jnp.where(V_next > V_thr, 1.0, 0.0)
        V_next = jnp.where(V_next > V_thr, V_reset, V_next)
        return x, V_next


class MlpBlock(nn.Module):
    """
        dense + LIF = 1 layer
        2 layers for 1 MLP block
    """
    features: int

    def setup(self):
        self.Dense1 = Dense(features=self.features)
        self.Dense2 = Dense(features=self.features)

    @nn.compact
    def __call__(self, x, V1, V2):
        x = self.Dense1(x)
        x, V1 = LIF()(x, V1)
        x = self.Dense2(x)
        x, V2 = LIF()(x, V2)
        return x, V1, V2


class MixerBlock(nn.Module):
    """
        tokens MLP block + channels MLP block = Mixer block layer
    """
    tokens_mlp_dim: int
    channels_mlp_dim: int

    def setup(self):
        self.MlpBlock1 = MlpBlock(self.tokens_mlp_dim, name='token_mixing')
        self.MlpBlock2 = MlpBlock(self.channels_mlp_dim, name='channel_mixing')
        self.layernorm = nn.LayerNorm()

    def __call__(self, x, V):
        # tokens mixer
        y = self.layernorm(x)
        y = jnp.swapaxes(y, 1, 2)
        V1 = jnp.swapaxes(V[0], 1, 2)
        V2 = jnp.swapaxes(V[1], 1, 2)
        y, V1, V2 = self.MlpBlock1(y, V1, V2)
        y = jnp.swapaxes(y, 1, 2)
        V1 = jnp.swapaxes(V1, 1, 2)
        V2 = jnp.swapaxes(V2, 1, 2)
        x = x + y
        # channels mixer
        y = self.layernorm(x)
        y, V3, V4 = self.MlpBlock2(y, V[2], V[3])
        x = x + y
        return x, jnp.concatenate(
            (jnp.expand_dims(V1, 0), jnp.expand_dims(V2, 0), jnp.expand_dims(V3, 0), jnp.expand_dims(V4, 0)), axis=0)


class MlpMixer(nn.Module):
    """
        Mixer architecture.
        for now 1 block only
    """
    patch_size: int
    num_classes: int
    time_seq: int
    image_size: int

    def setup(self):
        self.patch_num = int(self.image_size / self.patch_size)
        self.tokens_mlp_dim = self.patch_num * self.patch_num
        self.channels_mlp_dim = self.patch_size * self.patch_size * 3
        self.MixerBlock1 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock2 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock3 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock4 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock5 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock6 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock11 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock21 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock31 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock41 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock51 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.MixerBlock61 = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)
        self.Dense = Dense(self.num_classes, name='head')
        self.layernorm = nn.LayerNorm()

    def __call__(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], 3, self.image_size, self.image_size).transpose(0, 2, 3, 1)
        # cut into patches
        inputs = inputs.reshape(inputs.shape[0], self.patch_num, self.patch_size, self.patch_num, self.patch_size,
                           3).transpose(0, 1, 3, 2, 4, 5).reshape(inputs.shape[0], self.tokens_mlp_dim,
                                                                  self.channels_mlp_dim)
        # mlp mixer
        V1 = jnp.repeat(jnp.expand_dims(jnp.zeros_like(inputs), 0), 4, 0)
        V2 = jnp.zeros_like(V1)
        V3 = jnp.zeros_like(V1)
        V4 = jnp.zeros_like(V1)
        V5 = jnp.zeros_like(V1)
        V6 = jnp.zeros_like(V1)
        V11 = jnp.zeros_like(V1)
        V21 = jnp.zeros_like(V1)
        V31 = jnp.zeros_like(V1)
        V41 = jnp.zeros_like(V1)
        V51 = jnp.zeros_like(V1)
        V61 = jnp.zeros_like(V1)
        out = jnp.zeros((inputs.shape[0], self.num_classes))
        for _ in range(self.time_seq):
            x, V1 = self.MixerBlock1(inputs, V1)
            x, V2 = self.MixerBlock2(x, V2)
            x, V3 = self.MixerBlock3(x, V3)
            x, V4 = self.MixerBlock4(x, V4)
            x, V5 = self.MixerBlock5(x, V5)
            x, V6 = self.MixerBlock6(x, V6)
            x, V11 = self.MixerBlock11(x, V11)
            x, V21 = self.MixerBlock21(x, V21)
            x, V31 = self.MixerBlock31(x, V31)
            x, V41 = self.MixerBlock41(x, V41)
            x, V51 = self.MixerBlock51(x, V51)
            x, V61 = self.MixerBlock61(x, V61)
            # output
            x = self.layernorm(x)
            x = jnp.mean(x, axis=1)
            out += self.Dense(x)
        return out
