#ifndef _GPU_OPS_KERNELS_H_
#define _GPU_OPS_KERNELS_H_

#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace gpu_ops
{
    enum ElementType
    {
        F32,
        BF16,
    };

    struct MFMDescriptor
    {
        int numRows;
        int numCols;
        int batchSize;
        int miniBatchSize;
        int contextLen;
        ElementType f_type;
    };

    void gpu_BF16(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

} // namespace gpu_ops

#endif