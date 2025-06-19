import jax
import jax.numpy as jnp
from jax import vmap
#from ec.ops.bf16Vec_boolMat_kernel import mvm_p_fwd

def conn_dense(kernel, x):
    """Matmul for Bool 0/1 kernel and FP vector x"""

    # Check dtypes
    assert kernel.dtype == jnp.bool_, "Kernel must be boolean."
    assert x.dtype      != jnp.bool_, "Inputs must not be boolean."

    # matmul
    return jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))
'''
def conn_dense_ops(kernel, x):
    """Matmul for Bool 0/1 kernel and FP vector x"""

    # Check dtypes
    assert kernel.dtype == jnp.bool_, "Kernel must be boolean."
    assert x.dtype      != jnp.bool_, "Inputs must not be boolean."
   
   
    return vmap(mvm_p_fwd,in_axes=(0,None),out_axes=0)(x,kernel)[0]
'''