#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#endif

struct Half4 {
  half2 a;
  half2 b;
};


template <typename T>
struct LoadStore {
  static inline __device__ T load(void* p) {
    return *((T*) p);
  }

  static inline __device__ void store(void* p, const T& v) {
    *((T*) p) = v;
  }
};

template <>
struct LoadStore<Half4> {
  static inline __device__ Half4 load(void* p) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];" :
        "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b)) : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];" :
        "=r"(out.a.x), "=r"(out.b.x) : "l"(p));
#endif
    return out;
  }

  static inline __device__ void store(void* p, const Half4& vc) {
    Half4& v = const_cast<Half4&>(vc);
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p),
        "r"(__HALF2_TO_UI(v.a)), "r"(__HALF2_TO_UI(v.b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(v.a.x), "r"(v.b.x));
#endif
  }
};


template<typename T>
struct Vec4T {};

template<>
struct Vec4T<float> {
  using T = float4;
  using AccT = float4;
  static inline __device__ AccT zero() {
    float4 r;
    r.x = 0.0f;
    r.y = 0.0f;
    r.z = 0.0f;
    r.w = 0.0f;
    return r;
  }

  static inline __device__ AccT toAcc(const T& v) {
    return v;
  }
  static inline __device__ T fromAcc(const AccT& v) {
    return v;
  }
};

template<>
struct Vec4T<double> {
  using T = double4;
  using AccT = double4;

  static inline __device__ AccT zero() {
    double4 r;
    r.x = 0.0f;
    r.y = 0.0f;
    r.z = 0.0f;
    r.w = 0.0f;
    return r;
  }

  static inline __device__ AccT toAcc(const T& v) {
    return v;
  }
  static inline __device__ T fromAcc(const AccT& v) {
    return v;
  }
};

template<>
struct Vec4T<at::Half> {
  using T = Half4;
  using AccT = float4;

  static inline __device__ AccT zero() {
    float4 r;
    r.x = 0.0f;
    r.y = 0.0f;
    r.z = 0.0f;
    r.w = 0.0f;
    return r;
  }

  static inline __device__ AccT toAcc(const T& v) {
    float2 a = __half22float2(v.a);
    float2 b = __half22float2(v.b);
  
    float4 out;
    out.x = a.x;
    out.y = a.y;
    out.z = b.x;
    out.w = b.y;
  
    return out;
  }
  static inline __device__ T fromAcc(const AccT& v) {
    float2 a;
    a.x = v.x;
    a.y = v.y;
  
    float2 b;
    b.x = v.z;
    b.y = v.w;
  
    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
  
    return out;
  }

};


constexpr int L_max = 200;

static inline __device__ void gpuAtomicAdd(double* address, double val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(float* address, float val) {
  atomicAdd(address, val);
}

static inline __device__ void gpuAtomicAdd(at::Half* address, at::Half val) {
#if ((CUDA_VERSION < 10000) || \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    at::Half hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
#else
  atomicAdd(reinterpret_cast<__half*>(address), val);
#endif
}

// template <typename scalar_t>
// __global__ void sparse_embedding_cuda_forward_offsets_impl(
//     const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
//         weights, // [E][T][D]
//     const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
//         indices, // [N = B x T total indices,]
//     const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
//         offsets, // [B][T+1]
//     torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
//         output) {
//   const int B = offsets.size(0);
//   const int T_1 = offsets.size(1);
//   const int T = T_1 - 1;
//   const int D = weights.size(2);
//   const int E = weights.size(0);
//   // nnz: offsets[t][b+1]-offsets[t][b]
//   int b = blockIdx.x;
//   int t = blockIdx.y;
//   int d = threadIdx.x;
//   const int32_t indices_start = offsets[b][t];
//   const int32_t indices_end = offsets[b][t+1];
//   int L = indices_end - indices_start;

//   at::acc_type<scalar_t, true> sum = 0.0f;
//   #pragma unroll 8
//   for (int l = 0; l < L; ++l) {
//     // TODO: does ldg work better?
//     const int32_t offset = __ldg(&offsets[indices_start + l]);
//     const int32_t ind = __ldg(&indices[offset]);
//     // TODO: enable asserts.
//     // assert(ind >= 0);
//     // assert(ind < E);
//     sum += __ldg(&weights[ind][t][d]);
//   }
//   output[b][t][d] = sum;
// }


template <typename scalar_t, bool T_blocked>
__global__ void sparse_embedding_cuda_forward_offsets_kernel_impl(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
        indices, // [N = B x T total indices,]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
        offsets, // [B][T+1]
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {

  extern __shared__ int shmem_indices[];

  const int B = offsets.size(0);
  const int T_1 = offsets.size(1);
  const int T = T_1 - 1;
  const int D = weights.size(2);
  const int E = weights.size(0);
  if (!T_blocked) {
    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;

    const int32_t indices_start = offsets[b][t];
    const int32_t indices_end = offsets[b][t+1];
    int L = indices_end - indices_start;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();

    at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
    for (int l = 0; l < L; ++l) {
      sum += __ldg((&weights[shmem_indices[l]][t][0]) + d);
    }
    output[b][t][d] = sum;
  } else {
    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int L = 0;
    if (t < T) {
      const int32_t indices_start = offsets[b][t];
      const int32_t indices_end = offsets[b][t+1];
      L = indices_end - indices_start;
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L_max + i] = __ldg(&indices[indices_start + i]);
      }
    }
    __syncthreads();
    if (t < T) {
      at::acc_type<scalar_t, true> sum = 0.0f;
#pragma unroll 8
      for (int l = 0; l < L; ++l) {
        sum += __ldg((&weights[shmem_indices[t_t * L_max + l]][t][0]) + d);
      }
      output[b][t][d] = sum;
    }
  }
}

template <typename scalar_t, bool T_blocked>
__global__ void sparse_embedding_cuda_fused_backward_update_offsets_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits>
        indices, // [N = B x T total indices,]
    const torch::PackedTensorAccessor32<int32_t, 2, torch::RestrictPtrTraits>
        offsets, // [B][T+1]
    float lr) {
  extern __shared__ int shmem_indices[];

  const int B = offsets.size(0);
  const int T_1 = offsets.size(1);
  const int T = T_1 - 1;
  const int D = weights.size(2);
  const int E = weights.size(0);
  
  if (!T_blocked) {
    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;

    const int32_t indices_start = offsets[b][t];
    const int32_t indices_end = offsets[b][t+1];
    int L = indices_end - indices_start;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
#pragma unroll 8
    for (int l = 0; l < L; ++l) {
      const auto g = __ldg(&grad_output[b][t][d]);
      *((&weights[shmem_indices[l]][t][0]) + d) -= lr * g;
    }
  } else {
    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    int L = 0;
    if (t < T) {
      const int32_t indices_start = offsets[b][t];
      const int32_t indices_end = offsets[b][t+1];
      L = indices_end - indices_start;
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L_max + i] = __ldg(&indices[indices_start + i]);
      }
    }
    __syncthreads();
    if (t < T) {
#pragma unroll 8
      for (int l = 0; l < L; ++l) {
        const auto g = __ldg(&grad_output[b][t][d]);
        *((&weights[shmem_indices[t_t * L_max + l]][t][0]) + d) -= lr * g;
      }
    }
  }
}

template <typename scalar_t>
__global__ void sparse_embedding_cuda_forward_kernel_impl(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {
  const int B = indices.size(0);
  const int T = indices.size(1);
  const int D = weights.size(2);
  const int L = indices.size(2);
  const int E = weights.size(0);

  int t_d = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int d = t_d % D;
  int t = t_d / D;
  if (t < T) {
    at::acc_type<scalar_t, true> sum = 0.0f;
    for (int l = 0; l < L; ++l) {
      // TODO: does ldg work better?
      const int32_t ind = __ldg(&indices[b][t][l]);
      // TODO: enable asserts.
      // assert(ind >= 0);
      // assert(ind < E);
      sum += __ldg(&weights[ind][t][d]);
    }
    output[b][t][d] = sum;
  }
}

template <typename scalar_t, bool T_blocked, bool vec4>
__global__ void sparse_embedding_cuda_forward_fast_kernel_impl(
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        indices,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        output) {
  extern __shared__ int shmem_indices[];
  if (!T_blocked) {
    const int L = indices.size(2);
    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;

    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg((&indices[b][t][0]) + i);
    }
    __syncthreads();
    if (!vec4) {
      at::acc_type<scalar_t, true> sum = 0.0f;
  #pragma unroll 8
      for (int l = 0; l < L; ++l) {
        sum += __ldg((&weights[shmem_indices[l]][t][0]) + d);
      }
      output[b][t][d] = sum;
    } else {
      typedef typename Vec4T<scalar_t>::T  VecT;
      auto sum = Vec4T<scalar_t>::zero();
      #pragma unroll 8
      for (int l = 0; l < L; ++l) {
        VecT val = LoadStore<VecT>::load((void*)((&weights[shmem_indices[l]][t][0]) + d * 4));
        auto vval = Vec4T<scalar_t>::toAcc(val);
        sum.x += vval.x;
        sum.y += vval.y;
        sum.z += vval.z;
        sum.w += vval.w;
      }
      auto ssum = Vec4T<scalar_t>::fromAcc(sum);
      LoadStore<VecT>::store((void*)((&output[b][t][0]) + d * 4), ssum);
    }
  } else {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < T) {
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L + i] = __ldg((&indices[b][t][0]) + i);
      }
    }
    __syncthreads();
    if (t < T) {
      if (!vec4) {
        at::acc_type<scalar_t, true> sum = 0.0f;
  #pragma unroll 8
        for (int l = 0; l < L; ++l) {
          sum += __ldg((&weights[shmem_indices[t_t * L + l]][t][0]) + d);
        }
        output[b][t][d] = sum;
      } else {
        typedef typename Vec4T<scalar_t>::T  VecT;
        auto sum = Vec4T<scalar_t>::zero();
        #pragma unroll 8
        for (int l = 0; l < L; ++l) {
          VecT val = LoadStore<VecT>::load((void*)((&weights[shmem_indices[t_t * L + l]][t][0]) + d * 4));
          auto vval = Vec4T<scalar_t>::toAcc(val);
          sum.x += vval.x;
          sum.y += vval.y;
          sum.z += vval.z;
          sum.w += vval.w;
        }
        auto ssum = Vec4T<scalar_t>::fromAcc(sum);
        LoadStore<VecT>::store((void*)((&output[b][t][0]) + d * 4), ssum);
      }      
    }
  }
}

template <typename scalar_t>
__global__ void sparse_embedding_cuda_fused_backward_update_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        indices,
    float lr) {
  const int B = indices.size(0);
  const int T = indices.size(1);
  const int D = weights.size(2);
  const int L = indices.size(2);
  const int E = weights.size(0);
  // grad_output: [B][T][D]
  // weights: [E][T][D]
  // indices: [B][T][L]
  // launch grid of size [B][T][L x D]

  // what if we launch a kernel for every !grad_output - collaboratively load
  // indices L for each index, and then scatter the update to each l?
  // That is, [d] threads per block

  int l_d = blockIdx.x * blockDim.x + threadIdx.x;
  int b = blockIdx.y;
  int t = blockIdx.z;
  int d = l_d % D;
  int l = l_d / D;
  if (l < L) {
    const auto g = __ldg(&grad_output[b][t][d]);
    const int32_t ind = __ldg(&indices[b][t][l]);
    // TODO: need atomicadd?
    // gpuAtomicAdd(&weights[ind][t][d], -lr * g);
    weights[ind][t][d] -= lr * g;
  }
}

template <typename scalar_t, bool T_blocked, bool vec4>
__global__ void sparse_embedding_cuda_fused_backward_update_fast_kernel_impl(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>
        grad_output,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits>
        weights,
    const torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits>
        indices,
    float lr) {
  extern __shared__ int shmem_indices[];

  if (!T_blocked) {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int b = blockIdx.x;
    int t = blockIdx.y;
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[i] = __ldg((&indices[b][t][0]) + i);
    }
    __syncthreads();

#pragma unroll 8

    for (int l = 0; l < L; ++l) {     
      if (!vec4)  {
        const auto g = __ldg((&grad_output[b][t][0]) + d);
        *((&weights[shmem_indices[l]][t][0]) + d) -= lr * g;
      } else {
        typedef typename Vec4T<scalar_t>::T  VecT;
        auto g = LoadStore<VecT>::load((void*)((&grad_output[b][t][0]) + d * 4));
        auto gg = Vec4T<scalar_t>::toAcc(g);
        auto w = LoadStore<VecT>::load((void*)((&weights[shmem_indices[l]][t][0]) + d * 4));
        auto ww = Vec4T<scalar_t>::toAcc(w);
        ww.x -= gg.x * lr;
        ww.y -= gg.y * lr;
        ww.z -= gg.z * lr;
        ww.w -= gg.w * lr;
        LoadStore<VecT>::store((void*)((&weights[shmem_indices[l]][t][0]) + d * 4), Vec4T<scalar_t>::fromAcc(ww));
      }
    }
  } else {
    const int L = indices.size(2);
    const int T = indices.size(1);

    int d = threadIdx.x;
    int t_t = threadIdx.y;
    int b = blockIdx.x;
    int t_b = blockIdx.y;
    int t = blockIdx.y * blockDim.y + threadIdx.y;

    // [T_t][L] tile.
    // need T_t * L <= 32
    if (t < T) {
      for (int i = threadIdx.x; i < L; i += blockDim.x) {
        shmem_indices[t_t * L + i] = __ldg((&indices[b][t][0]) + i);
      }
    }
    __syncthreads();

    if (t < T) {
      #pragma unroll 8
      for (int l = 0; l < L; ++l) {
        if (!vec4) {
          const auto g = __ldg((&grad_output[b][t][0]) + d);
          *((&weights[shmem_indices[t_t * L + l]][t][0]) + d) -= lr * g;
        } else {
          typedef typename Vec4T<scalar_t>::T  VecT;
          auto g = LoadStore<VecT>::load((void*)((&grad_output[b][t][0]) + d * 4));
          auto gg = Vec4T<scalar_t>::toAcc(g);
          auto w = LoadStore<VecT>::load((void*)((&weights[shmem_indices[t_t * L + l]][t][0]) + d * 4));
          auto ww = Vec4T<scalar_t>::toAcc(w);
          ww.x -= gg.x * lr;
          ww.y -= gg.y * lr;
          ww.z -= gg.z * lr;
          ww.w -= gg.w * lr;
          LoadStore<VecT>::store((void*)((&weights[shmem_indices[t_t * L + l]][t][0]) + d * 4), Vec4T<scalar_t>::fromAcc(ww));
        }
      }
    }
  }
}

torch::Tensor sparse_embedding_cuda_forward_kernel(
    // [E][T][D]
    torch::Tensor weights,
    // [B][T][L // #device]
    torch::Tensor indices) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);

  auto output = at::empty({B, T, D}, weights.options());

  const int threads = 1024;
  const dim3 blocks((T * D + threads - 1) / threads, B);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_forward", ([&] {
        sparse_embedding_cuda_forward_kernel_impl<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
      }));
  AT_CUDA_CHECK(cudaGetLastError());

  return output;
}

torch::Tensor sparse_embedding_cuda_forward_fast_kernel(
    // [E][T][D]
    torch::Tensor weights,
    // [B][T][L // #device]
    torch::Tensor indices) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const auto L = indices.size(2);
  auto output = at::empty({B, T, D}, weights.options());


  if (D <= 256) {
    const int T_t = 256 / D;
    const int T_b = (T + T_t - 1) / T_t;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward", ([&] {
          if (std::is_same<scalar_t, at::Half>::value) {
            const dim3 threads(D / 4, T_t);
            AT_ASSERT(D % 4 == 0);
            const dim3 blocks(B, T_b);
        
            sparse_embedding_cuda_forward_fast_kernel_impl<
                scalar_t, true, true><<<blocks, threads, T_t * L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
          } else {
            const dim3 threads(D, T_t);
            const dim3 blocks(B, T_b);
        
            sparse_embedding_cuda_forward_fast_kernel_impl<
                scalar_t, true, false><<<blocks, threads, T_t * L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());

          }
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward", ([&] {
          if (std::is_same<scalar_t, at::Half>::value) {
            const int threads = D / 4;
            AT_ASSERT(D % 4 == 0);
            const dim3 blocks(B, T);
        
            sparse_embedding_cuda_forward_fast_kernel_impl<scalar_t, false, true><<<
                blocks, threads, L * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
          } else {
            const int threads = D;
            const dim3 blocks(B, T);
            sparse_embedding_cuda_forward_fast_kernel_impl<scalar_t, false, false><<<
                blocks, threads, L * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
          }
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return output;
}


torch::Tensor sparse_embedding_cuda_forward_offsets_kernel(
  // [E][T][D]
  torch::Tensor weights,
  // [N = \sum_B \sum_T L_{b, t}]
  torch::Tensor indices, 
  // [B][T+1]
  torch::Tensor offsets) {
  const auto B = offsets.size(0);
  const auto T_1 = offsets.size(1);
  const auto T = T_1 - 1;
  const auto D = weights.size(2);
  auto output = at::empty({B, T, D}, weights.options());
  if (D < 64) {
    const int T_t = std::min<int>(64 / D, 4);
    const int T_b = (T + T_t - 1) / T_t;
    const dim3 threads(D, T_t);
    const dim3 blocks(B, T_b);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward_offsets", ([&] {
          sparse_embedding_cuda_forward_offsets_kernel_impl<
              scalar_t, true><<<blocks, threads, T_t * L_max * sizeof(int),
                                at::cuda::getCurrentCUDAStream()>>>(
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {
    const int threads = D;
    const dim3 blocks(B, T);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_forward_offsets", ([&] {
          sparse_embedding_cuda_forward_offsets_kernel_impl<scalar_t, false><<<
              blocks, threads, L_max * sizeof(int), at::cuda::getCurrentCUDAStream()>>>(
              weights
                  .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
              output
                  .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>());
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return output;
}


void sparse_embedding_cuda_backward_update_kernel(torch::Tensor grad_output,
                                                  torch::Tensor weights,
                                                  torch::Tensor indices,
                                                  float lr) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const int L = indices.size(2);

  const int threads = 1024;
  const dim3 blocks((L * D + threads - 1) / threads, B, T);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_backward_update", ([&] {
        sparse_embedding_cuda_fused_backward_update_kernel_impl<scalar_t><<<
            blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_output
                .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
            lr);
      }));
  AT_CUDA_CHECK(cudaGetLastError());

  return;
}

void sparse_embedding_cuda_backward_update_fast_kernel(
    torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices,
    float lr) {
  const auto B = indices.size(0);
  const auto T = indices.size(1);
  const auto D = weights.size(2);
  const auto L = indices.size(2);
  if (D < 256) {
    const int T_t = 256 / D;
    const int T_b = (T + T_t - 1) / T_t;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_backward_update_fast", ([&] {
          if (std::is_same<at::Half, scalar_t>::value) {
            const dim3 threads(D / 4, T_t);
            AT_ASSERT(D % 4 == 0);

            const dim3 blocks(B, T_b);
            sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
                scalar_t, true, true><<<blocks, threads, T_t * L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                grad_output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                lr);
          } else {
            const dim3 threads(D, T_t);
            const dim3 blocks(B, T_b);
            sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
                scalar_t, true, false><<<blocks, threads, T_t * L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                grad_output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                lr);
          }
      }));
    AT_CUDA_CHECK(cudaGetLastError());
  } else {

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        weights.type(), "sparse_embedding_cuda_backward_update_fast", ([&] {
          if (std::is_same<at::Half, scalar_t>::value) {
            AT_ASSERT(D % 4 == 0);
            const int threads = D / 4;
            const dim3 blocks(B, T);
            sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
                scalar_t, false, true><<<blocks, threads, L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                grad_output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                lr);
          } else {
            const int threads = D;
            const dim3 blocks(B, T);
            sparse_embedding_cuda_fused_backward_update_fast_kernel_impl<
                scalar_t, false, false><<<blocks, threads, L * sizeof(int),
                                  at::cuda::getCurrentCUDAStream()>>>(
                grad_output
                    .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                weights
                    .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                indices.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>(),
                lr);
          }
        }));
    AT_CUDA_CHECK(cudaGetLastError());
  }
  return;
}

void sparse_embedding_cuda_backward_update_offsets_kernel(
  torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices, torch::Tensor offsets,
  float lr) {

const auto B = offsets.size(0);
const auto T_1 = offsets.size(1);
const auto T = T_1 - 1;
const auto D = weights.size(2);
  
if (D < 64) {
  const int T_t = std::min<int>(64 / D, 4);
  const int T_b = (T + T_t - 1) / T_t;
  const dim3 threads(D, T_t);
  const dim3 blocks(B, T_b);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_backward_update_offsets", ([&] {
        sparse_embedding_cuda_fused_backward_update_offsets_kernel_impl<
            scalar_t, true><<<blocks, threads, T_t * L_max * sizeof(int),
                              at::cuda::getCurrentCUDAStream()>>>(
            grad_output
                .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights
                .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            lr);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
} else {
  const int threads = D;
  const dim3 blocks(B, T);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "sparse_embedding_cuda_backward_update_offsets", ([&] {
        sparse_embedding_cuda_fused_backward_update_offsets_kernel_impl<
            scalar_t, false><<<blocks, threads, L_max * sizeof(int),
                               at::cuda::getCurrentCUDAStream()>>>(
            grad_output
                .packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            weights
                .packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
            indices.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
            offsets.packed_accessor32<int32_t, 2, torch::RestrictPtrTraits>(),
            lr);
      }));
  AT_CUDA_CHECK(cudaGetLastError());
}
return;
}