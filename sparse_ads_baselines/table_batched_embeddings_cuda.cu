#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cub/device/device_radix_sort.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <mutex>

using namespace at;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))

namespace {

static constexpr int32_t kWarpSize = 32;
static constexpr int32_t kMaxThreads = 1024;

template <typename T> struct Vec4T {};

struct Half4 {
  half2 a;
  half2 b;

  __device__ inline void store(Half *p) {
#if CUDA_VERSION >= 9000
    asm("st.v2.u32 [%0], {%1, %2};"
        :
        : "l"(p), "r"(__HALF2_TO_UI(a)), "r"(__HALF2_TO_UI(b)));
#else
    asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(a.x), "r"(b.x));
#endif
  }
};

template <> struct Vec4T<Half> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const Half *p) {
    Half4 out;
#if CUDA_VERSION >= 9000
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
        : "l"(p));
#else
    asm("ld.global.v2.u32 {%0, %1}, [%2];"
        : "=r"(out.a.x), "=r"(out.b.x)
        : "l"(p));
#endif

    float2 a = __half22float2(out.a);
    float2 b = __half22float2(out.b);

    acc.x = a.x;
    acc.y = a.y;
    acc.z = b.x;
    acc.w = b.y;
  }

  DEVICE_INLINE void store(Half *p) {
    float2 a;
    a.x = acc.x;
    a.y = acc.y;

    float2 b;
    b.x = acc.z;
    b.y = acc.w;

    Half4 out;
    out.a = __float22half2_rn(a);
    out.b = __float22half2_rn(b);
    out.store(p);
  }
};

template <> struct Vec4T<float> {
  float4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const float *p) { acc = *((const float4 *)p); }

  DEVICE_INLINE void store(float *p) { *((float4 *)p) = acc; }
};

template <> struct Vec4T<double> {
  double4 acc;
  DEVICE_INLINE Vec4T() {
    acc.x = 0;
    acc.y = 0;
    acc.z = 0;
    acc.w = 0;
  }

  DEVICE_INLINE Vec4T(const double *p) { acc = *((const double4 *)p); }

  DEVICE_INLINE void store(double *p) { *((double4 *)p) = acc; }
};

template <typename T>
DEVICE_INLINE T shfl_xor(const T val, int laneMask, int width = kWarpSize) {
#if CUDA_VERSION >= 9000
  return __shfl_xor_sync(0xffffffff, val, laneMask, width);
#else
  return __shfl_xor(val, laneMask, width);
#endif
}

/// Sums a register value across all warp threads
template <typename T, int ReduceWidth = kWarpSize>
DEVICE_INLINE T warpReduceAllSum(T val) {
#pragma unroll
  for (int mask = ReduceWidth / 2; mask > 0; mask >>= 1) {
    val += shfl_xor(val, mask);
  }
  return val;
}

static DEVICE_INLINE double gpuAtomicAdd(double *address, double val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE float gpuAtomicAdd(float *address, float val) {
  return atomicAdd(address, val);
}

static DEVICE_INLINE at::Half gpuAtomicAdd(at::Half *address, at::Half val) {
#if ((CUDA_VERSION < 10000) ||                                                 \
     (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)))
  unsigned int *address_as_ui =
      (unsigned int *)((char *)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  at::Half hsum;
  do {
    assumed = old;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
  hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
  return hsum;
#else
  return atomicAdd(reinterpret_cast<__half *>(address), val);
#endif
}
}

template <typename scalar_t> struct UnweightedForward {
  DEVICE_INLINE void accumulate(Vec4T<scalar_t> &sum, Vec4T<scalar_t> weight,
                                int32_t indices_offset) {
    sum.acc.x += weight.acc.x;
    sum.acc.y += weight.acc.y;
    sum.acc.z += weight.acc.z;
    sum.acc.w += weight.acc.w;
  }
};

template <typename scalar_t> struct WeightedForward {
  const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> indice_weights_;

  WeightedForward(const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
                      indice_weights)
      : indice_weights_(indice_weights) {}

  DEVICE_INLINE void accumulate(Vec4T<scalar_t> &sum, Vec4T<scalar_t> weight,
                                int32_t indices_offset) {
    acc_type<scalar_t, true> element_weight = indice_weights_[indices_offset];
    sum.acc.x += weight.acc.x * element_weight;
    sum.acc.y += weight.acc.y * element_weight;
    sum.acc.z += weight.acc.z * element_weight;
    sum.acc.w += weight.acc.w * element_weight;
  }
};

template <typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_1(
    // [\sum_t E_t][D]
    const PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [T][B][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [T x B + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [T][.
    PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits>
        output, // [B][T][D],
    int32_t L_max,
    F f) {

  extern __shared__ int32_t shmem_indices[];

  const int32_t B = output.size(0);
  const int32_t T = output.size(1);
  const int32_t D = output.size(2);

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_indices) {
    int32_t shmem_offset = threadIdx.y * L_max;
    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        Vec4T<scalar_t> weight((&weights[table_offset + idx][0]) + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][t][0]) + d * 4);
    }
  }
}

Tensor batched_embedding_forward_cuda(Tensor weights, Tensor table_offsets,
                                      Tensor indices, Tensor offsets,
                                      c10::optional<Tensor> indice_weights,
                                      int64_t L_max, int64_t BT_block_size,
                                      bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size != 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT((B * T) % BT_block_size == 0);
  AT_ASSERT(D % 4 == 0);
  const dim3 threads(std::min(D / 4, kMaxThreads / BT_block_size),
                     BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);

#define X(shmem_param, shmem_size, functor)                                    \
  batched_embedding_forward_kernel_1<scalar_t, (shmem_param)><<<               \
      blocks, threads, (shmem_size), at::cuda::getCurrentCUDAStream()>>>(      \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),              \
      static_cast<int32_t>(L_max), (functor))

  auto output = empty({B, T, D}, weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_forward_kernel", ([&] {
        if (shmem) {
          if (indice_weights) {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              UnweightedForward<scalar_t>());
          }
        } else {
          if (indice_weights) {
            X(false, 0,
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(false, 0, UnweightedForward<scalar_t>());
          }
        }
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, bool use_atomics> struct SGDFunctor {
  SGDFunctor(float learning_rate) : learning_rate_(learning_rate) {}
  float learning_rate_;

  inline void __device__ operator()(scalar_t *weight, Vec4T<scalar_t> grad) {
    // don't handle atomic case.
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * learning_rate_;
    weight_new.acc.y -= grad.acc.y * learning_rate_;
    weight_new.acc.z -= grad.acc.z * learning_rate_;
    weight_new.acc.w -= grad.acc.w * learning_rate_;
    weight_new.store(weight);
  }
};

template <typename scalar_t, bool shared_mem, typename F>
__global__ void batched_embedding_backward_sgd_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    int32_t L_max, F f) {

  extern __shared__ int32_t shmem_indices[];

  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_mem) {
    int32_t shmem_offset = threadIdx.y * L_max;

    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();

    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        f(&weights[table_offset + idx][0] + d * 4, g);
      }
    }
  } else {
    for (int32_t l = 0; l < L; ++l) {
      auto idx = __ldg(&indices[indices_start + l]);
      for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
        Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
        f(&weights[table_offset + idx][0] + d * 4, g);
      }
    }
  }
}

void batched_embedding_backward_sgd_cuda(Tensor grad_output, Tensor weights,
                                         Tensor table_offsets, Tensor indices,
                                         Tensor offsets, float learning_rate,
                                         int64_t L_max, int64_t BT_block_size,
                                         bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(BT_block_size > 0);
  AT_ASSERT(D % 4 == 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_kernel", ([&] {
        const dim3 threads(std::min(D / 4, kMaxThreads / BT_block_size),
                           BT_block_size);
        const dim3 blocks((B * T) / BT_block_size);

        if (shmem) {
          batched_embedding_backward_sgd_kernel_1<
              scalar_t,
              true><<<blocks, threads, BT_block_size * L_max * sizeof(int32_t),
                      at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, false>(learning_rate));
        } else {
          batched_embedding_backward_sgd_kernel_1<
              scalar_t,
              false><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
              grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),
              weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),
              table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
              static_cast<int32_t>(L_max),
              SGDFunctor<scalar_t, false>(learning_rate));
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());
  return;
}

template <typename scalar_t, bool use_atomics> struct AdaGradFunctor {
  AdaGradFunctor(float learning_rate, float eps)
      : learning_rate_(learning_rate), eps_(eps) {}
  float learning_rate_;
  float eps_;
  DEVICE_INLINE acc_type<scalar_t, true>
  update_momentum(acc_type<scalar_t, true> sum_square_grads,
                  acc_type<scalar_t, true> *optimizer_state,
                  int32_t indices_offset) {
    acc_type<scalar_t, true> old_sum_square_grads;
    if (use_atomics) {
      old_sum_square_grads = gpuAtomicAdd(optimizer_state, sum_square_grads);
    } else {
      old_sum_square_grads = __ldg(optimizer_state);
      *optimizer_state = old_sum_square_grads + sum_square_grads;
    }
    return learning_rate_ *
           (1.0 / (sqrt(old_sum_square_grads + sum_square_grads) + eps_));
  }

  struct State {};
  DEVICE_INLINE State init_update(int thread_id) { return State{}; }

  DEVICE_INLINE void update_weight(scalar_t *weight,
                                   acc_type<scalar_t, true> multiplier,
                                   Vec4T<scalar_t> grad, State &state,
                                   State &sample_weight_state) {
    // can't use atomics.
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    weight_new.store(weight);
  }

  DEVICE_INLINE State init_sample_weight(int32_t indices_offset) {
    return State{};
  }

  DEVICE_INLINE void update_sample_weight(State &state,
                                          int32_t indices_offset) {
    return;
  }
};

template <typename scalar_t, bool use_atomics> struct WeightedAdaGradFunctor {
  WeightedAdaGradFunctor(
      float learning_rate, float eps,
      const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          indice_weights,
      PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          grad_indice_weights)
      : learning_rate_(learning_rate), eps_(eps),
        indice_weights_(indice_weights),
        grad_indice_weights_(grad_indice_weights) {}

  float learning_rate_;
  float eps_;
  const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> indice_weights_;
  PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> grad_indice_weights_;

  DEVICE_INLINE acc_type<scalar_t, true>
  update_momentum(acc_type<scalar_t, true> sum_square_grads,
                  acc_type<scalar_t, true> *optimizer_state,
                  int32_t indices_offset) {
    acc_type<scalar_t, true> element_weight = indice_weights_[indices_offset];
    acc_type<scalar_t, true> weighted_sum_square_grads =
        sum_square_grads * element_weight * element_weight;
    acc_type<scalar_t, true> old_sum_square_grads;
    if (use_atomics) {
      old_sum_square_grads =
          gpuAtomicAdd(optimizer_state, weighted_sum_square_grads);
    } else {
      old_sum_square_grads = __ldg(optimizer_state);
      *optimizer_state = old_sum_square_grads + weighted_sum_square_grads;
    }
    return element_weight * learning_rate_ *
           (1.0 /
            (sqrt(old_sum_square_grads + weighted_sum_square_grads) + eps_));
  }

  struct State {};
  DEVICE_INLINE State init_update(int thread_id) { return State{}; }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, State &state,
                acc_type<scalar_t, true> &sample_weight_state) {
    // can't use atomics.
    Vec4T<scalar_t> weight_new(weight);

    sample_weight_state +=
        weight_new.acc.x * grad.acc.x + weight_new.acc.y * grad.acc.y +
        weight_new.acc.z * grad.acc.z + weight_new.acc.w * grad.acc.w;

    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    weight_new.store(weight);
  }
  DEVICE_INLINE acc_type<scalar_t, true>
  init_sample_weight(int32_t indices_offset) {
    return 0.0;
  }

  DEVICE_INLINE void
  update_sample_weight(acc_type<scalar_t, true> &sample_weight_state,
                       int32_t indices_offset) {
    auto accumulated_sample_weight_state =
        warpReduceAllSum<acc_type<scalar_t, true>>(sample_weight_state);
    // one thread per warp responsible for updating parameter.
    // TODO: ugly?
    if (threadIdx.x == 0) {
      grad_indice_weights_[indices_offset] = accumulated_sample_weight_state;
    }
  }
};

template <typename scalar_t>
DEVICE_INLINE void stochastic_rounding_vector(scalar_t *output,
                                              Vec4T<scalar_t> value,
                                              uint4 random_bits) {
  value.store(output);
}

template <>
DEVICE_INLINE void stochastic_rounding_vector(Half *output, Vec4T<Half> value,
                                              uint4 random_bits) {
  Half4 v;

  v.a.x = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.x) + (random_bits.x >> 19)));
  v.a.y = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.y) + (random_bits.y >> 19)));
  v.b.x = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.z) + (random_bits.z >> 19)));
  v.b.y = __float2half_rz(
      __uint_as_float(__float_as_uint(value.acc.w) + (random_bits.w >> 19)));
  v.store(output);
}

template <typename scalar_t, bool use_atomics>
struct StochasticRoundingWeightedAdaGradFunctor
    : public WeightedAdaGradFunctor<scalar_t, use_atomics> {
  StochasticRoundingWeightedAdaGradFunctor(
      float learning_rate, float eps,
      const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          indice_weights,
      PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits>
          grad_indice_weights,
      std::pair<uint64_t, uint64_t> seeds)
      : WeightedAdaGradFunctor<scalar_t, use_atomics>(
            learning_rate, eps, indice_weights, grad_indice_weights),
        seeds_(seeds) {}
  std::pair<uint64_t, uint64_t> seeds_;

  DEVICE_INLINE curandStatePhilox4_32_10_t init_update(int thread_id) {
    curandStatePhilox4_32_10_t state;
    curand_init(seeds_.first, thread_id, seeds_.second, &state);
    return state;
  }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, curandStatePhilox4_32_10_t &state,
                acc_type<scalar_t, true> &sample_weight_state) {
    Vec4T<scalar_t> weight_new(weight);
    sample_weight_state +=
        weight_new.acc.x * grad.acc.x + weight_new.acc.y * grad.acc.y +
        weight_new.acc.z * grad.acc.z + weight_new.acc.w * grad.acc.w;
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    uint4 bits = curand4(&state);
    stochastic_rounding_vector(weight, weight_new, bits);
  }
};

template <typename scalar_t, bool use_atomics>
struct StochasticRoundingAdaGradFunctor
    : public AdaGradFunctor<scalar_t, use_atomics> {
  StochasticRoundingAdaGradFunctor(float learning_rate, float eps,
                                   std::pair<uint64_t, uint64_t> seeds)
      : AdaGradFunctor<scalar_t, use_atomics>(learning_rate, eps),
        seeds_(seeds) {}
  std::pair<uint64_t, uint64_t> seeds_;

  DEVICE_INLINE curandStatePhilox4_32_10_t init_update(int thread_id) {
    curandStatePhilox4_32_10_t state;
    curand_init(seeds_.first, thread_id, seeds_.second, &state);
    return state;
  }

  DEVICE_INLINE void
  update_weight(scalar_t *weight, acc_type<scalar_t, true> multiplier,
                Vec4T<scalar_t> grad, curandStatePhilox4_32_10_t &state,
                typename AdaGradFunctor<scalar_t, use_atomics>::State
                    &sample_weight_state) {
    Vec4T<scalar_t> weight_new(weight);
    weight_new.acc.x -= grad.acc.x * multiplier;
    weight_new.acc.y -= grad.acc.y * multiplier;
    weight_new.acc.z -= grad.acc.z * multiplier;
    weight_new.acc.w -= grad.acc.w * multiplier;
    uint4 bits = curand4(&state);
    stochastic_rounding_vector(weight, weight_new, bits);
  }
};

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_approx_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem[];
  auto *shmem_multipliers = (acc_type<scalar_t, true> *)(&shmem[0]);
  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> g(&grad_output[b][t][0] + d * 4);
    g_local_sum_square += g.acc.x * g.acc.x + g.acc.y * g.acc.y +
                          g.acc.z * g.acc.z + g.acc.w * g.acc.w;
  }
  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_multipliers[threadIdx.y * L_max + i] = f.update_momentum(
        g_sum_square, &optimizer_state[table_offset + idx], indices_start + i);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    auto idx = indices[indices_start + l];
    acc_type<scalar_t, true> multiplier =
        shmem_multipliers[threadIdx.y * L_max + l];

    auto sample_weight_state = f.init_sample_weight(indices_start + l);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      f.update_weight(&weights[table_offset + idx][0] + d * 4, multiplier,
                      grad_out, state, sample_weight_state);
    }
    f.update_sample_weight(sample_weight_state, indices_start + l);
  }
}

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(D % 4 == 0);
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);
  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights = at::empty(indices.sizes(), grad_output.options());
  }

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_approx_kernel_1<                          \
      scalar_t><<<blocks, threads,                                             \
                  BT_block_size * L_max * sizeof(acc_type<scalar_t, true>),    \
                  at::cuda::getCurrentCUDAStream()>>>(                         \
      grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      static_cast<int32_t>(L_max), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            auto f = WeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>());
            X(f);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
                L_max * ((D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            auto f = StochasticRoundingWeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                rng_engine_inputs);
            X(f);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

static void CUDAManagedDeleter(void *ptr) { AT_CUDA_CHECK(cudaFree(ptr)); }

struct CUDAManagedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaMallocManaged(&ptr, size));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation,
                                cudaCpuDeviceId));
    AT_CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy,
                                at::cuda::current_device()));
    return {ptr,
            ptr,
            &CUDAManagedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override { return &CUDAManagedDeleter; }
};

static CUDAManagedAllocator g_managed_allocator;

static void CUDAHostMappedDeleter(void *ptr) {
  AT_CUDA_CHECK(cudaFreeHost(ptr));
}

struct CUDAHostMappedAllocator final : public at::Allocator {
  at::DataPtr allocate(size_t size) const override {
    void *ptr;
    AT_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocWriteCombined |
                                                cudaHostAllocMapped));

    void *dev_ptr;
    AT_CUDA_CHECK(cudaHostGetDevicePointer(&dev_ptr, ptr, 0));
    return {dev_ptr,
            ptr,
            &CUDAHostMappedDeleter,
            {at::DeviceType::CUDA, at::cuda::current_device()}};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &CUDAHostMappedDeleter;
  }
};

static CUDAHostMappedAllocator g_host_mapped_allocator;

std::vector<int64_t> defaultStrides(IntArrayRef sizes) {
  std::vector<int64_t> strides(sizes.size());
  int64_t stride = 1;
  for (size_t i = sizes.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= sizes[i - 1];
  }
  return strides;
}

int64_t computeStorageSize(IntArrayRef sizes, IntArrayRef strides) {
  // size of the underlying storage is 1 bigger than the offset
  // of the last element according to stride
  int64_t size = 1;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (sizes[i] == 0) {
      return 0;
    }
    size += strides[i] * (sizes[i] - 1);
  }
  return size;
}

Tensor new_managed_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(Storage::use_byte_size_t(), computeStorageSize(sizes, strides),
                         &g_managed_allocator,
                         /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

Tensor new_host_mapped_tensor(Tensor self, std::vector<std::int64_t> sizes) {
  auto strides = defaultStrides(sizes);
  auto storage = Storage(Storage::use_byte_size_t(), computeStorageSize(sizes, strides),
                         &g_host_mapped_allocator,
                         /*resizable=*/false);
  auto tensor = at::empty({0}, self.options()).set_(storage, 0, sizes, strides);
  return tensor;
}

template <typename scalar_t, bool shared_indices, typename F>
__global__ void batched_embedding_forward_kernel_mixed_D_1(
    // [\sum_t E_t x D_t]
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>
        table_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices, // [N = \sum_{b,t} L_{b,t} total indices, i.e. flattened
                 // [B][T][L]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        offsets, // [B x T + 1]
    // offsets = cumsum([0] + lengths.contiguous()), where lengths L is [B][T].
    PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits>
        output, // [B][\sum_t D_t],
    int32_t L_max,
    F f) {

  extern __shared__ int32_t shmem_indices[];
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t t = b_t / B;
  int32_t b = b_t % B;
  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  const int64_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  if (shared_indices) {
    int32_t shmem_offset = threadIdx.y * L_max;
    for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
      shmem_indices[shmem_offset + i] = __ldg(&indices[indices_start + i]);
    }
    __syncthreads();
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = shmem_indices[shmem_offset + l];
        Vec4T<scalar_t> weight((&weights[0]) + table_offset + idx * D + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][0]) + dim_start + d * 4);
    }
  } else {
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> sum;
      for (int32_t l = 0; l < L; ++l) {
        auto idx = __ldg(&indices[indices_start + l]);
        Vec4T<scalar_t> weight((&weights[0]) + table_offset + idx * D + d * 4);
        f.accumulate(sum, weight, indices_start + l);
      }
      sum.store((&output[b][0]) + dim_start + d * 4);
    }
  }
}

Tensor batched_embedding_forward_mixed_D_cuda(
    Tensor weights, Tensor table_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    int64_t L_max, int64_t BT_block_size, bool shmem) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size != 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT((B * T) % BT_block_size == 0);
  const dim3 threads(kMaxThreads / BT_block_size, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);

#define X(shmem_param, shmem_size, functor)                                    \
  batched_embedding_forward_kernel_mixed_D_1<scalar_t, (shmem_param)><<<       \
      blocks, threads, (shmem_size), at::cuda::getCurrentCUDAStream()>>>(      \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),              \
      static_cast<int32_t>(L_max), (functor))

  auto output = empty({B, total_D}, weights.options());
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_forward_kernel", ([&] {
        if (shmem) {
          if (indice_weights) {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(true, BT_block_size * L_max * sizeof(int32_t),
              UnweightedForward<scalar_t>());
          }
        } else {
          if (indice_weights) {
            X(false, 0,
              WeightedForward<scalar_t>(
                  indice_weights
                      ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>()));
          } else {
            X(false, 0, UnweightedForward<scalar_t>());
          }
        }
      }));

#undef X
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_approx_kernel_mixed_D_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]

    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    int32_t L_max, F f) {
  extern __shared__ int32_t shmem[];
  auto *shmem_multipliers = (acc_type<scalar_t, true> *)(&shmem[0]);
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> g(&grad_output[b][0] + dim_start + d * 4);
    g_local_sum_square += g.acc.x * g.acc.x + g.acc.y * g.acc.y +
                          g.acc.z * g.acc.z + g.acc.w * g.acc.w;
  }
  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  const int32_t table_dim_offset = table_dim_offsets[t];

  const int64_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);
  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    shmem_multipliers[threadIdx.y * L_max + i] = f.update_momentum(
        g_sum_square, &optimizer_state[table_dim_offset + idx],
        indices_start + i);
  }
  __syncthreads();
  for (int32_t l = 0; l < L; ++l) {
    auto idx = indices[indices_start + l];
    acc_type<scalar_t, true> multiplier =
        shmem_multipliers[threadIdx.y * L_max + l];

    auto sample_weight_state = f.init_sample_weight(indices_start + l);
    for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      f.update_weight((&weights[0]) + table_offset + idx * D + d * 4,
                      multiplier, grad_out, state, sample_weight_state);
    }
    f.update_sample_weight(sample_weight_state, indices_start + l);
  }
}

c10::optional<Tensor> batched_embedding_backward_adagrad_approx_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps, int64_t L_max,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);
  const dim3 blocks((B * T) / BT_block_size);
  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights =
        at::empty(indice_weights->sizes(), grad_output.options());
  }

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_approx_kernel_mixed_D_1<                  \
      scalar_t><<<blocks, threads,                                             \
                  BT_block_size * L_max * sizeof(acc_type<scalar_t, true>),    \
                  at::cuda::getCurrentCUDAStream()>>>(                         \
      grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),    \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      static_cast<int32_t>(L_max), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            auto f = WeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>());
            X(f);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
                L_max * ((total_D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            auto f = StochasticRoundingWeightedAdaGradFunctor<scalar_t, false>(
                learning_rate, eps,
                indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                grad_indice_weights
                    ->packed_accessor32<scalar_t, 1, RestrictPtrTraits>(),
                rng_engine_inputs);
            X(f);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

__global__ void construct_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 2, RestrictPtrTraits>
        batch_offsets_per_table,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        total_indices_per_table,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> output) {
  const int32_t T = batch_offsets_per_table.size(0);
  const int32_t B = batch_offsets_per_table.size(1);

  // do warp-per-D (so only need warp reduction)
  int32_t b_t = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t b = b_t % B;
  int32_t t = b_t / B;
  if (t < T) {
    int32_t upper = 0;
    if (b != B - 1) {
      upper = batch_offsets_per_table[t][b + 1];
    } else {
      upper = total_indices_per_table[t];
    }
    int32_t lower = batch_offsets_per_table[t][b];
    output[1 + t * B + b] = upper - lower;
  }
}

Tensor construct_offsets(Tensor batch_offsets_per_table, // [T][B]
                         Tensor total_indices_per_table  // [T]
                         ) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(batch_offsets_per_table.get_device());

  const auto T = batch_offsets_per_table.size(0);
  const auto B = batch_offsets_per_table.size(1);
  const dim3 threads(kMaxThreads);
  const dim3 blocks((B * T + 1 + kMaxThreads - 1) / kMaxThreads);

  Tensor output =
      at::zeros({1 + B * T}, batch_offsets_per_table.options().dtype(kInt));

  construct_offsets_kernel<<<blocks, threads, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
      batch_offsets_per_table
          .packed_accessor32<int32_t, 2, RestrictPtrTraits>(),
      total_indices_per_table
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      output.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

struct IndexInfo {
  int32_t b_t;
  int32_t indices_idx;
};

__global__ void linear_index_weight_offsets_mixed_D_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    IndexInfo *__restrict__ infos,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> linear_indices) {
  const int32_t T = table_dim_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_dim_offset = table_dim_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    IndexInfo info;
    info.b_t = b_t;
    info.indices_idx = indices_start + i;
    infos[indices_start + i] = info;
    linear_indices[indices_start + i] = table_dim_offset + idx;
  }
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_exact_kernel_mixed_D_1(
    const PackedTensorAccessor32<scalar_t, 2, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        table_dim_offsets, // [T]
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        dim_offsets, // [T+1]

    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos,

    int32_t indices_numel, F f) {
  const int32_t T = table_offsets.size(0) - 1;
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  const int32_t t = info_0.b_t / B;
  const int64_t table_offset = table_offsets[t];
  const int32_t table_dim_offset = table_dim_offsets[t];
  const int32_t idx = linear_index - table_dim_offset;
  const int32_t dim_start = dim_offsets[t];
  const int32_t dim_end = dim_offsets[t + 1];
  const int32_t D = dim_end - dim_start;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    g_local_sum_square += sum.acc.x * sum.acc.x + sum.acc.y * sum.acc.y +
                          sum.acc.z * sum.acc.z + sum.acc.w * sum.acc.w;
  }

  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);

  acc_type<scalar_t, true> multiplier = f.update_momentum(
      g_sum_square, &optimizer_state[table_dim_offset + idx], -1);

  auto sample_weight_state = f.init_sample_weight(-1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      auto b = infos[sorted_linear_indice_id + sl].b_t % B;
      Vec4T<scalar_t> grad_out(&grad_output[b][0] + dim_start + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    f.update_weight((&weights[0]) + table_offset + idx * D + d * 4, multiplier,
                    sum, state, sample_weight_state);
  }
  f.update_sample_weight(sample_weight_state, -1);
}

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_mixed_D_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets,
    Tensor table_dim_offsets, Tensor dim_offsets, int64_t total_D,
    Tensor indices, Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0) - 1;
  AT_ASSERT(T > 0);
  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);

  const dim3 threads(kWarpSize, BT_block_size);

  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights =
        at::empty(indice_weights->sizes(), grad_output.options());
  }

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  linear_index_weight_offsets_mixed_D_kernel<<<
      dim3((B * T) / BT_block_size), threads, 0,
      at::cuda::getCurrentCUDAStream()>>>(
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks((linear_indices.numel() + BT_block_size - 1) /
                    BT_block_size);

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_exact_kernel_mixed_D_1<                   \
      scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(     \
      grad_output.packed_accessor32<scalar_t, 2, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),        \
      table_dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),    \
      dim_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),          \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      linear_indices_sorted                                                    \
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),                 \
      (const IndexInfo *)infos_sorted.data_ptr(),                              \
      static_cast<int32_t>(linear_indices_sorted.numel()), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            rng_engine_inputs = at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(
                ((total_D + kWarpSize - 1) / kWarpSize));
          }
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
                learning_rate, eps, rng_engine_inputs);
            X(f);
          }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}

__global__ void linear_index_weight_offsets_kernel(
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    IndexInfo *__restrict__ infos,
    PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> linear_indices) {
  const int32_t T = table_offsets.size(0);
  const int32_t B = (offsets.size(0) - 1) / T;

  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  int32_t b = b_t % B;
  int32_t t = b_t / B;

  const int32_t table_offset = table_offsets[t];
  const int32_t indices_start = offsets[t * B + b];
  const int32_t indices_end = offsets[t * B + b + 1];
  int32_t L = indices_end - indices_start;

  for (int32_t i = threadIdx.x; i < L; i += blockDim.x) {
    auto idx = __ldg(&indices[indices_start + i]);
    IndexInfo info;
    info.b_t = b_t;
    info.indices_idx = indices_start + i;
    infos[indices_start + i] = info;
    linear_indices[indices_start + i] = table_offset + idx;
  }
}

template <typename scalar_t, typename F>
__global__ void batched_embedding_backward_adagrad_exact_kernel_1(
    const PackedTensorAccessor32<scalar_t, 3, RestrictPtrTraits> grad_output,
    PackedTensorAccessor64<scalar_t, 2, RestrictPtrTraits> weights,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_offsets,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> offsets,
    PackedTensorAccessor32<acc_type<scalar_t, true>, 1, RestrictPtrTraits>
        optimizer_state,
    const PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
        sorted_linear_indices,
    const IndexInfo *__restrict__ infos, int32_t indices_numel, F f) {
  const int32_t B = grad_output.size(0);
  const int32_t T = grad_output.size(1);
  const int32_t D = grad_output.size(2);

  int32_t sorted_linear_indice_id = blockIdx.x * blockDim.y + threadIdx.y;
  if (sorted_linear_indice_id >= indices_numel) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  // check if this warp is responsible for this whole segment.
  bool segment_start = (sorted_linear_indice_id == 0 ||
                        sorted_linear_indices[sorted_linear_indice_id - 1] !=
                            sorted_linear_indices[sorted_linear_indice_id]);

  if (!segment_start) {
    // don't have *warp* divergence since we launch full warps in blockDim.x, so
    // we can just exit this warp entirely.
    return;
  }

  int32_t linear_index = sorted_linear_indices[sorted_linear_indice_id];
  // now, find the end of the segment (and thus the segment length `SL`).
  int32_t SL = 1;
  while (sorted_linear_indice_id + SL < indices_numel &&
         sorted_linear_indices[sorted_linear_indice_id + SL] == linear_index) {
    SL += 1;
  }
  // now, each segment corresponds to exactly one table `t` and row in that
  // table (`idx`). Thus, we can hoist out some of the book-keeping.
  auto info_0 = infos[sorted_linear_indice_id];
  const int32_t t = info_0.b_t / B;
  const int64_t table_offset = table_offsets[t];
  const int32_t idx = linear_index - table_offset;

  acc_type<scalar_t, true> g_local_sum_square = 0;
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      int32_t b = infos[sorted_linear_indice_id + sl].b_t % B;
      // TODO: apply per-sample weighting.
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    g_local_sum_square += sum.acc.x * sum.acc.x + sum.acc.y * sum.acc.y +
                          sum.acc.z * sum.acc.z + sum.acc.w * sum.acc.w;
  }

  const acc_type<scalar_t, true> g_sum_square =
      warpReduceAllSum<acc_type<scalar_t, true>>(g_local_sum_square);

  auto state = f.init_update(blockIdx.x * blockDim.x * blockDim.y +
                             threadIdx.y * blockDim.x + threadIdx.x);

  acc_type<scalar_t, true> multiplier =
      f.update_momentum(g_sum_square, &optimizer_state[table_offset + idx], -1);

  auto sample_weight_state = f.init_sample_weight(-1);
  for (int32_t d = threadIdx.x; d * 4 < D; d += blockDim.x) {
    Vec4T<scalar_t> sum;
    for (int32_t sl = 0; sl < SL; ++sl) {
      auto b = infos[sorted_linear_indice_id + sl].b_t % B;
      Vec4T<scalar_t> grad_out(&grad_output[b][t][0] + d * 4);
      sum.acc.x += grad_out.acc.x;
      sum.acc.y += grad_out.acc.y;
      sum.acc.z += grad_out.acc.z;
      sum.acc.w += grad_out.acc.w;
    }
    f.update_weight(&weights[table_offset + idx][0] + d * 4, multiplier, sum,
                    state, sample_weight_state);
  }
  f.update_sample_weight(sample_weight_state, -1);
}

c10::optional<Tensor> batched_embedding_backward_adagrad_exact_cuda(
    Tensor grad_output, Tensor weights, Tensor table_offsets, Tensor indices,
    Tensor offsets, c10::optional<Tensor> indice_weights,
    Tensor optimizer_state, float learning_rate, float eps,
    bool stochastic_rounding, int64_t BT_block_size) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());

  const auto T = table_offsets.size(0);
  AT_ASSERT(T > 0);
  const auto D = weights.size(1);
  AT_ASSERT(D > 0);

  // offsets = [B x T  + 1]
  const auto B = (offsets.size(0) - 1) / T;
  AT_ASSERT(B > 0);
  AT_ASSERT(BT_block_size > 0);
  if ((B * T) % BT_block_size != 0) {
    BT_block_size = 1;
  }
  AT_ASSERT(D % 4 == 0);
  AT_ASSERT(BT_block_size * kWarpSize <= kMaxThreads);
  const dim3 threads(kWarpSize, BT_block_size);

  c10::optional<Tensor> grad_indice_weights = c10::nullopt;
  if (indice_weights) {
    grad_indice_weights = at::empty(indices.sizes(), grad_output.options());
  }

  auto infos =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto infos_sorted =
      at::empty({indices.numel() * static_cast<int64_t>(sizeof(IndexInfo))},
                indices.options().dtype(kByte));

  auto linear_indices = at::empty_like(indices);
  auto linear_indices_sorted = at::empty_like(indices);

  linear_index_weight_offsets_kernel<<<dim3((B * T) / BT_block_size), threads,
                                       0, at::cuda::getCurrentCUDAStream()>>>(
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
      (IndexInfo *)infos.data_ptr(),
      linear_indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>());
  AT_CUDA_CHECK(cudaGetLastError());

  size_t temp_storage_bytes = 0;
  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      nullptr, temp_storage_bytes, (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  auto temp_storage = at::empty({static_cast<int64_t>(temp_storage_bytes)},
                                indices.options().dtype(kByte));

  AT_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_storage_bytes,
      (int32_t *)linear_indices.data_ptr(),
      (int32_t *)linear_indices_sorted.data_ptr(),
      (IndexInfo *)infos.data_ptr(), (IndexInfo *)infos_sorted.data_ptr(),
      linear_indices.numel(), 0, int(log2(optimizer_state.numel()) + 1),
      at::cuda::getCurrentCUDAStream(), false));

  const dim3 blocks((linear_indices.numel() + BT_block_size - 1) /
                    BT_block_size);

#define X(functor)                                                             \
  batched_embedding_backward_adagrad_exact_kernel_1<                           \
      scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(     \
      grad_output.packed_accessor32<scalar_t, 3, RestrictPtrTraits>(),         \
      weights.packed_accessor64<scalar_t, 2, RestrictPtrTraits>(),             \
      table_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),        \
      indices.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),              \
      optimizer_state.packed_accessor32<acc_type<scalar_t, true>, 1,           \
                                        RestrictPtrTraits>(),                  \
      linear_indices_sorted                                                    \
          .packed_accessor32<int32_t, 1, RestrictPtrTraits>(),                 \
      (const IndexInfo *)infos_sorted.data_ptr(),                              \
      static_cast<int32_t>(linear_indices_sorted.numel()), (functor))

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weights.type(), "batched_embedding_backward_adagrad_approx_kernel", ([&] {
        if (!stochastic_rounding) {
          if (indice_weights) {
            AT_ASSERT(false);
          } else {
            auto f = AdaGradFunctor<scalar_t, false>(learning_rate, eps);
            X(f);
          }
        } else {
          std::pair<uint64_t, uint64_t> rng_engine_inputs;
          {
            auto gen = at::cuda::detail::getDefaultCUDAGenerator();
            std::lock_guard<std::mutex> lock(gen.mutex());
            rng_engine_inputs =
                at::check_generator<at::CUDAGeneratorImpl>(gen)->philox_engine_inputs(((D + kWarpSize - 1) / kWarpSize));
          }
          // if (indice_weights) {
          //   AT_ASSERT(false);
          // } else {
          //   auto f = StochasticRoundingAdaGradFunctor<scalar_t, false>(
          //       learning_rate, eps, rng_engine_inputs);
          //   X(f);
          // }
        }
      }));
  AT_CUDA_CHECK(cudaGetLastError());

#undef X

  return grad_indice_weights;
}
