#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm> // 包含std::fill函数
#include "kernel_helpers.h"
#include "kernels.h"
#include <cublas_v2.h>
using namespace std;

#define BLOCK_SIZE 16
namespace gpu_ops
{

    void ThrowIfError(cudaError_t error)
    {
        if (error != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(error));
        }
    }

    template <typename T>
    __global__ void MFMKernel(const T *__restrict__ A, const bool *__restrict__ B, T *__restrict__ C, int ARows, int ACols, int BCols)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < ARows && col < BCols)
        {
            T sum = 0;
#pragma unroll
            for (int k = 0; k < ACols; ++k)
            {
                if (B[k * BCols + col])
                {
                    sum += A[row * ACols + k];
                }
            }
            C[row * BCols + col] = sum;
        }
    }

    template <typename T>
    __global__ void BatchMFMKernel(const T *__restrict__ A, const bool *__restrict__ B, T *__restrict__ C, int batchSize, int ARows, int ACols, int BCols)
    {
        int batch = blockIdx.z;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch < batchSize && row < ARows && col < BCols)
        {
            T sum = 0;
#pragma unroll
            for (int k = 0; k < ACols; ++k)
            {
                if (B[(batch * ACols * BCols) + (k * BCols) + col])
                {
                    sum += A[(batch * ARows * ACols) + (row * ACols) + k];
                }
            }
            C[(batch * ARows * BCols) + (row * BCols) + col] = sum;
        }
    }

    template <typename T>
    __global__ void tensorMulKernel(const T *__restrict__ A, const bool *__restrict__ B, T *__restrict__ C, int batch_size, int dim1, int dim2, int dim3, int dim4)
    {

        int i = blockIdx.x;
        int j = blockIdx.y;
        int k = threadIdx.x;
        int m = threadIdx.y;

        T sum = 0;

        for (int l = 0; l < dim3; l++)
        {
            if (B[i * dim3 * dim4 + l * dim4 + m])
            {
                sum += A[i * dim1 * dim2 * dim3 + j * dim2 * dim3 + k * dim3 + l];
            }
        }

        C[i * dim1 * dim2 * dim4 + j * dim2 * dim4 + k * dim4 + m] = sum;
    }

    void gpu_BF16(cudaStream_t stream, void **buffers, const char *opaque,
                  std::size_t opaque_len)
    {
        const MFMDescriptor &d = *UnpackDescriptor<MFMDescriptor>(opaque, opaque_len);

        const int M = d.batchSize; // batchSize
        const int K = d.numRows;   // inputDim
        const int N = d.numCols;   // outputDim
        const int miniBatchSize = d.miniBatchSize;
        const int contextLen = d.contextLen;

        const __nv_bfloat16 *x_dev = reinterpret_cast<const __nv_bfloat16 *>(buffers[0]);
        const bool *Weight_dev = reinterpret_cast<const bool *>(buffers[1]);
        __nv_bfloat16 *y_dev = reinterpret_cast<__nv_bfloat16 *>(buffers[2]);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

        if (miniBatchSize != 0 && contextLen == 0) // 3d
        {
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                         (M + blockDim.y - 1) / blockDim.y,
                         M);
            BatchMFMKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(x_dev, Weight_dev, y_dev, M, miniBatchSize, K, N);
        }
        else if (miniBatchSize != 0 && contextLen != 0) // 4d - 3d
        {

            dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                         (M + blockDim.y - 1) / blockDim.y,
                         M);
            for (int i = 0; i < miniBatchSize; ++i)
            {
                BatchMFMKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(x_dev + i * (M * contextLen * K), Weight_dev, y_dev + i * (M * contextLen * N), M, contextLen, K, N);
            }
        }
        else if (miniBatchSize == 0 && contextLen == 0) // 2d
        {
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                         (M + blockDim.y - 1) / blockDim.y);
            MFMKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(x_dev, Weight_dev, y_dev, M, K, N);
        }
        else if (miniBatchSize == 0 && contextLen != 0) // 3d - 2d
        {
            dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                         (M + blockDim.y - 1) / blockDim.y);
            for (int i = 0; i < M; ++i)
            {
                MFMKernel<__nv_bfloat16><<<gridDim, blockDim, 0, stream>>>(x_dev + i * (contextLen * K), Weight_dev, y_dev + i * (contextLen * N), contextLen, K, N);
            }
           
        }

                ThrowIfError(cudaGetLastError());
    }
}