// Fused RMSNorm kernel.
//
// PyTorch version is three ops (pow+mean, rsqrt+mul, mul-by-weight), each a
// separate kernel launch that reads the activation from HBM and writes it back.
// This does it in one kernel: one read of x, one write of y.
//
// Layout: one thread block per row (one token's hidden vector), 256 threads.
// Each thread sums squares over its strided slice of the row, a warp-shuffle
// + shared-memory reduction produces the row's mean-of-squares, then each
// thread writes y = x * rsqrt(ms + eps) * w for its slice.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define THREADS 256

template <typename T>
__device__ __forceinline__ float to_float(T v);
template <> __device__ __forceinline__ float to_float<float>(float v) { return v; }
template <> __device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <> __device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <> __device__ __forceinline__ float from_float<float>(float v) { return v; }
template <> __device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

template <typename T>
__global__ void rmsnorm_kernel(const T* __restrict__ x, const T* __restrict__ w,
                               T* __restrict__ y, int dim, float eps) {
    const int row = blockIdx.x;
    const T* xr = x + (size_t)row * dim;
    T* yr = y + (size_t)row * dim;

    // 1. partial sum of squares over this thread's slice
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += THREADS) {
        float v = to_float<T>(xr[i]);
        ss += v * v;
    }

    // 2. reduce across the block: warp shuffle, then across warps via shared mem
    __shared__ float warp_sums[THREADS / 32];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < THREADS / 32) ? warp_sums[threadIdx.x] : 0.f;
        v = warp_sum(v);
        if (threadIdx.x == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(warp_sums[0] / dim + eps);

    // 3. write normalized output
    for (int i = threadIdx.x; i < dim; i += THREADS) {
        float v = to_float<T>(xr[i]) * inv * to_float<T>(w[i]);
        yr[i] = from_float<T>(v);
    }
}

torch::Tensor rmsnorm_forward(torch::Tensor x, torch::Tensor w, double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    auto y = torch::empty_like(x);
    const int dim = x.size(-1);
    const int rows = x.numel() / dim;
    auto wc = w.to(x.dtype()).contiguous();

    AT_DISPATCH_SWITCH(x.scalar_type(), "rmsnorm",
        AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
            rmsnorm_kernel<float><<<rows, THREADS>>>(x.data_ptr<float>(), wc.data_ptr<float>(), y.data_ptr<float>(), dim, (float)eps);
        })
        AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
            rmsnorm_kernel<__half><<<rows, THREADS>>>((const __half*)x.data_ptr<at::Half>(), (const __half*)wc.data_ptr<at::Half>(), (__half*)y.data_ptr<at::Half>(), dim, (float)eps);
        })
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
            rmsnorm_kernel<__nv_bfloat16><<<rows, THREADS>>>((const __nv_bfloat16*)x.data_ptr<at::BFloat16>(), (const __nv_bfloat16*)wc.data_ptr<at::BFloat16>(), (__nv_bfloat16*)y.data_ptr<at::BFloat16>(), dim, (float)eps);
        })
    );
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rmsnorm_forward, "fused RMSNorm forward");
}
