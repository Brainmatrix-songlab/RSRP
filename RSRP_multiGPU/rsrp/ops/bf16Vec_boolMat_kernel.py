from jax import core, dtypes
from jax.lib import xla_client
from jax.core import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call

from functools import partial

from ec.ops.build import gpu_ops

def mvm_p_fwd(vector, matrix):
    output = mvm_p.bind(vector, matrix)
    return output

for _name, _value in gpu_ops.get_mvm_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): gpu_ops.ElementType.BF16,
        ir.F32Type.get(): gpu_ops.ElementType.F32,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def _mvm_p_fwd_cuda_lowering(ctx, vector,matrix):
    vector_type = ir.RankedTensorType(vector.type)
    vector_shape = vector_type.shape 
    matrix_type = ir.RankedTensorType(matrix.type)
    matrix_shape = matrix_type.shape 

    if len(vector_shape) ==2:
        batchSize,inputDim = vector_shape
        inputDim,outPutDim = matrix_shape
        result_shape = (batchSize, outPutDim)

        opaque = gpu_ops.create_mvm_descriptor(
            inputDim,
            outPutDim,
            batchSize,
            0,
            0,
            element_type_to_descriptor_type_mapping(vector_type.element_type),
        )

        out = custom_call(
            b"gpu_BF16",
            result_types=[
                ir.RankedTensorType.get(result_shape, vector_type.element_type), 
            ], 
            operands=[vector,matrix], 
            backend_config=opaque,
            operand_layouts=default_layouts(vector_shape,matrix_shape), 
            result_layouts=default_layouts(result_shape),
        ).results
        return out
        
    elif len(vector_shape) == 3 and len(matrix_shape) == 3: 
        batchSize,miniBatchSize,inputDim = vector_shape
        batchSize,inputDim,outPutDim = matrix_shape
        result_shape = (batchSize, miniBatchSize,outPutDim)

        opaque = gpu_ops.create_mvm_descriptor(
            inputDim,
            outPutDim,
            batchSize,
            miniBatchSize,
            0,
            element_type_to_descriptor_type_mapping(vector_type.element_type),
        )

        out = custom_call(
            b"gpu_BF16",
            result_types=[
                ir.RankedTensorType.get(result_shape, vector_type.element_type), 
            ],
            operands=[vector,matrix], 
            backend_config=opaque,
            operand_layouts=default_layouts(vector_shape,matrix_shape),
            result_layouts=default_layouts(result_shape),
        ).results
        return out
    elif len(vector_shape) == 3 and len(matrix_shape) == 2: 
        batchSize,contextLen,inputDim = vector_shape
        inputDim,outPutDim =  matrix_shape
        result_shape = (batchSize,contextLen,outPutDim) 

        opaque = gpu_ops.create_mvm_descriptor(
            inputDim,
            outPutDim,
            batchSize,
            0,
            contextLen,
            element_type_to_descriptor_type_mapping(vector_type.element_type),
        )

        out = custom_call(
            b"gpu_BF16",
            result_types=[
                ir.RankedTensorType.get(result_shape, vector_type.element_type), 
            ],
            operands=[vector,matrix], 
            backend_config=opaque,
            operand_layouts=default_layouts(vector_shape,matrix_shape),
            result_layouts=default_layouts(result_shape),
        ).results
        return out
    
    elif len(vector_shape) == 4 and len(matrix_shape) == 3:  
        batchSize,miniBatchSize,contextLen,inputDim = vector_shape
        batchSize,inputDim,outPutDim =  matrix_shape
        result_shape = (batchSize,miniBatchSize,contextLen,outPutDim)  

        opaque = gpu_ops.create_mvm_descriptor(
            inputDim,
            outPutDim,
            batchSize,
            miniBatchSize,
            contextLen,
            element_type_to_descriptor_type_mapping(vector_type.element_type),
        )

        out = custom_call(
            b"gpu_BF16",
            result_types=[
                ir.RankedTensorType.get(result_shape, vector_type.element_type), 
            ],
            operands=[vector,matrix], 
            backend_config=opaque,
            operand_layouts=default_layouts(vector_shape,matrix_shape),
            result_layouts=default_layouts(result_shape),
        ).results
        return out
   

  
def _mvm_p_fwd_abstract(vector,matrix):
    w_dtype = dtypes.canonicalize_dtype(vector.dtype)
    if len(vector.shape) == 2 and len(matrix.shape) == 2:
        batchSize,inputDim = vector.shape
        inputDim,outPutDim =  matrix.shape
        res_shape = (batchSize,outPutDim)    
    elif len(vector.shape) == 3 and len(matrix.shape) == 3:    
        batchSize,miniBatchSize,inputDim = vector.shape
        batchSize,inputDim,outPutDim =  matrix.shape
        res_shape = (batchSize,miniBatchSize,outPutDim)
    elif len(vector.shape) == 3 and len(matrix.shape) == 2:    
        batchSize,miniBatchSize,inputDim = vector.shape
        inputDim,outPutDim =  matrix.shape
        res_shape = (batchSize,miniBatchSize,outPutDim) 
    elif len(vector.shape) == 4 and len(matrix.shape) == 3:    
        batchSize,miniBatchSize,contextLen,inputDim = vector.shape
        batchSize,inputDim,outPutDim =  matrix.shape
        res_shape = (batchSize,miniBatchSize,contextLen,outPutDim)     
    return (
        ShapedArray(res_shape, w_dtype, named_shape=matrix.named_shape),  # output
    )


mvm_p = core.Primitive("mvm_p_fwd")
mvm_p.multiple_results = True
mvm_p.def_impl(partial(xla.apply_primitive, mvm_p))
mvm_p.def_abstract_eval(_mvm_p_fwd_abstract)

mlir.register_lowering(
    mvm_p,
    _mvm_p_fwd_cuda_lowering,
    platform="gpu",
)

def mvm_p_fwd_batch(args, axes):
    res = mvm_p_fwd(*args)
    return res,(0,)

batching.primitive_batchers[mvm_p] = mvm_p_fwd_batch