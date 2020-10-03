#include <torch/extension.h>

#include <vector>

#include <ATen/core/functional.h>
#include <torch/csrc/cuda/device_set.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/utils/hash.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <THC/THC.h>

#include "mpi.h"

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace cuda {
namespace nccl {

using namespace at;

namespace detail {

void throw_nccl_error(ncclResult_t status) {
  std::ostringstream err;
  err << "NCCL Error " << status << ": " << ncclGetErrorString(status);
  throw std::runtime_error(err.str());
}

struct NcclCommList {
  std::unique_ptr<ncclComm_t[]> comms;
  int ndevices;
  NcclCommList(const std::vector<int>& devices)
      : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
    NCCL_CHECK(ncclCommInitAll(comms.get(), devices.size(), devices.data()));
  }
  NcclCommList(NcclCommList&& foo) = default;
  ~NcclCommList() {
    /*
* TODO(T30279827) Temporarily disable calling ncclCommDestroy
* Calling ncclCommDestroy while program exiting is undefined
* according to Nvidia, and lead to segfault in NCCL 2
* (whether it is called before or after the CUDA runtime destructor).
* Temporarily disable it in destructor to avoid segfault.
* Following up with Nvidia for long term solution.
*/
    return;

    if (comms) {
      for (int i = 0; i < ndevices; i++) {
        int dummy_var;
        if (cudaGetDevice(&dummy_var) != cudaSuccess) {
          /* there are cases when this destructor is called after the
CUDA driver is already unloaded from the process.
In these cases, skip ncclCommDestroy */
          return;
        }
        ncclCommDestroy(comms[i]);
      }
    }
  }
  ArrayRef<ncclComm_t> ref() const {
    return ArrayRef<ncclComm_t>(comms.get(), ndevices);
  }
};

using device_list = std::vector<int>;
// accesses to this object have to be guarded by THC's CudaFreeMutex
static std::unordered_map<device_list, NcclCommList, torch::hash<device_list> >
    _communicators;

ArrayRef<ncclComm_t> get_communicators(TensorList inputs) {
  static auto get_device = [](const at::Tensor& t) -> int {
    return t.get_device();
  };
  device_list devices = fmap(inputs, get_device);
  auto it = _communicators.find(devices);
  if (it == _communicators.end())
    std::tie(it, std::ignore) = _communicators.emplace(devices, devices);
  return it->second.ref();
}

ncclDataType_t get_data_type(const Tensor& t) {
  if (t.type().backend() != Backend::CUDA) {
    throw std::runtime_error("Unconvertible NCCL type");
  }
  switch (t.scalar_type()) {
    case at::kFloat:
      return ncclFloat;
    case at::kHalf:
      return ncclHalf;
    case at::kDouble:
      return ncclDouble;
    case at::kLong:
      return ncclInt64;
    case at::kInt:
      return ncclInt;
    case at::kChar:
      return ncclChar;
    case at::kByte:
      return ncclChar;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

void check_inputs(TensorList inputs, TensorList outputs, int input_multiplier,
                  int output_multiplier) {
  // len(inputs) == len(outputs)
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but "
           "got input of length "
        << len << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto type = inputs[0].type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];
    auto output = outputs[i];

    if (!(input.is_cuda() && !input.is_sparse() && output.is_cuda() &&
          !output.is_sparse())) {
      throw std::runtime_error(
          "input and output elements have to be cuda dense Tensors");
    }

    if (!(type == input.type() && type == output.type())) {
      throw std::runtime_error(
          "all inputs and outputs must be of the same Tensor type");
    }

    if (!input.is_contiguous() || !output.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);

    // inputs and outputs must be on same device respectively
    if (input_device != output.get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    // all inputs must be same size
    if (input.numel() != numel) {
      throw std::runtime_error(
          "all inputs must have the same number of elements");
    }

    if (output.numel() * output_multiplier != numel * input_multiplier) {
      throw std::runtime_error(
          "output must be of size input_size * size_multiplier");
    }
  }
}

}  // namespace detail
}
}
}

torch::Tensor sparse_embedding_cuda_forward_kernel(torch::Tensor weights,
                                                   torch::Tensor indices);

torch::Tensor sparse_embedding_cuda_forward_fast_kernel(
    // [E][T][D]
    torch::Tensor weights,
    // [B][T][L // #device]
    torch::Tensor indices);


torch::Tensor sparse_embedding_cuda_forward_offsets_kernel(
    torch::Tensor weights, torch::Tensor indices, torch::Tensor offsets);

void sparse_embedding_cuda_backward_update_offsets_kernel(
  torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices, torch::Tensor offsets,
  float lr);


void sparse_embedding_cuda_backward_update_kernel(torch::Tensor grad_output,
                                                  torch::Tensor weights,
                                                  torch::Tensor indices,
                                                  float lr);

std::vector<at::Tensor> sparse_embedding_cuda_forward(
    // [device][E // #device][T][D]
    std::vector<at::Tensor> sharded_weights,
    // [device][B][T][L // #device]
    std::vector<at::Tensor> scattered_indices) {
  // -> [device][B // #device][T][D]
  using namespace torch::cuda::nccl::detail;
  const int64_t num_devices = sharded_weights.size();
  std::vector<torch::Tensor> sharded_embeddings;
  // [device][B][T][D]
  std::vector<torch::Tensor> outputs;

  at::cuda::OptionalCUDAGuard device_guard;
  for (int d = 0; d < num_devices; ++d) {
    AT_ASSERT(sharded_weights[d].get_device() ==
              scattered_indices[d].get_device());
    device_guard.set_index(sharded_weights[d].get_device());
    sharded_embeddings.push_back(sparse_embedding_cuda_forward_kernel(
        sharded_weights[d], scattered_indices[d]));
    const auto B = scattered_indices[d].sizes()[0];
    const auto T = scattered_indices[d].sizes()[1];
    const auto D = sharded_weights[d].sizes()[2];
    AT_ASSERT(B % num_devices == 0);
    outputs.push_back(at::empty({B / num_devices, T, D}, at::kCUDA));
  }
  if (num_devices == 1) {
    return sharded_embeddings;
  }

  pybind11::gil_scoped_release no_gil;
  ncclDataType_t data_type = get_data_type(sharded_embeddings[0]);
  int64_t count = sharded_embeddings[0].numel() / num_devices;
  auto comms = get_communicators(sharded_embeddings);

  AutoNcclGroup nccl_group_guard;
  check_inputs(sharded_embeddings, outputs, 1, num_devices);
  for (int d = 0; d < num_devices; ++d) {
    int device = sharded_embeddings[d].get_device();
    device_guard.set_index(device);
    auto stream = at::cuda::getCurrentCUDAStream(device).stream();
    NCCL_CHECK(ncclReduceScatter(sharded_embeddings[d].data_ptr(),
                                 outputs[d].data_ptr(), count, data_type,
                                 ncclSum, comms[d], stream));
  }
  return outputs;
}

at::Tensor sparse_embedding_cuda_forward_single(
    // [E][T][D]
    at::Tensor weights,
    // [B][T // #device][[L]
    at::Tensor scattered_indices) {
  // -> [B][T // #device][D]
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  return sparse_embedding_cuda_forward_kernel(weights, scattered_indices);
}

at::Tensor sparse_embedding_cuda_forward_fast_single(
    // [E][T][D]
    at::Tensor weights,
    // [B][T // #device][[L]
    at::Tensor scattered_indices) {
  // -> [B][T // #device][D]
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  return sparse_embedding_cuda_forward_fast_kernel(weights, scattered_indices);
}

at::Tensor sparse_embedding_cuda_forward_offsets(
    // [E][T][D]
    at::Tensor weights,
    // [\sum_{0 <= b < B, 0 <= t < T} L_{b, t}]
    at::Tensor indices,
    // [B][T+1]
    at::Tensor offsets
) {
  // -> [B][T][D]
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  return sparse_embedding_cuda_forward_offsets_kernel(weights, indices, offsets);
}


void sparse_embedding_cuda_backward_update_fast_kernel(
    torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices,
    float lr);

void sparse_embedding_cuda_backward_update_single(torch::Tensor grad_output,
                                                  torch::Tensor weights,
                                                  torch::Tensor indices,
                                                  float lr) {
  // -> [B][T // #device][D]
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  sparse_embedding_cuda_backward_update_kernel(grad_output, weights, indices,
                                               lr);
}

void sparse_embedding_cuda_backward_update_fast_single(
    torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices,
    float lr) {
  // -> [B][T // #device][D]
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  sparse_embedding_cuda_backward_update_fast_kernel(grad_output, weights,
                                                    indices, lr);
}

void sparse_embedding_cuda_backward_update_offsets(
    torch::Tensor grad_output, torch::Tensor weights, torch::Tensor indices, torch::Tensor offsets,
    float lr) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weights.get_device());
  sparse_embedding_cuda_backward_update_offsets_kernel(grad_output, weights, indices, offsets, lr);
}


static std::pair<MPI_Comm*, int> sparse_embedding_comm() {
  static std::once_flag once;
  static MPI_Comm world_comm;
  static int world_size;

  std::call_once(once, [&] {
    // initialize CUDA contexts
    for (auto i = 0; i < c10::cuda::device_count(); ++i) {
      at::cuda::CUDAGuard device_guard(i);
      auto tensor = at::empty({10, 10}, at::kCUDA);
    }
    {
      auto op = MPI_Comm_dup(MPI_COMM_WORLD, &world_comm);
      AT_ASSERT(op == MPI_SUCCESS);
    }
    {
      auto op = MPI_Comm_size(world_comm, &world_size);
      AT_ASSERT(op == MPI_SUCCESS);
    }
  });
  return {&world_comm, world_size};
}

static std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
};

void sparse_embedding_cuda_forward_all2all(at::Tensor embeddings,
                                           // [B][T // devices][D]
                                           at::Tensor result) {
  // -> [B // #device][T][D]
  auto comm_and_size = sparse_embedding_comm();
  at::cuda::CUDAGuard device_guard(embeddings.get_device());
  auto stream =
      at::cuda::getCurrentCUDAStream(embeddings.get_device()).stream();
  AT_ASSERT(embeddings.is_contiguous());
  AT_ASSERT(result.is_contiguous());

  AT_ASSERT(embeddings.numel() == result.numel());
  AT_ASSERT(embeddings.scalar_type() == result.scalar_type());

  // Need to synchronize our current stream so the GPU memory is valid before
  // issuing our collective call.
  {
    pybind11::gil_scoped_release no_gil;
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
    auto op = MPI_Alltoall(
        embeddings.data_ptr(), embeddings.numel() / comm_and_size.second,
        mpiDatatype.at(embeddings.scalar_type()), result.data_ptr(),
        result.numel() / comm_and_size.second,
        mpiDatatype.at(result.scalar_type()), *(comm_and_size.first));
    AT_ASSERT(op == MPI_SUCCESS);
  }
}

static ncclComm_t nccl_comm() {
  using namespace torch::cuda::nccl::detail;
  static std::once_flag once;
  static ncclComm_t world_comm;

  std::call_once(once, [&] {
    auto comm_and_size = sparse_embedding_comm();
    int my_rank;
    {
      auto op = MPI_Comm_rank(*(comm_and_size.first), &my_rank);
      AT_ASSERT(op == MPI_SUCCESS);
    }

    ncclUniqueId id;
    // generating NCCL unique ID at one process and broadcasting it to all
    if (my_rank == 0) {
      ncclGetUniqueId(&id);
    }
    {
      auto op = MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0,
                          *(comm_and_size.first));
      AT_ASSERT(op == MPI_SUCCESS);
    }
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(
        ncclCommInitRank(&world_comm, comm_and_size.second, id, my_rank));
    NCCL_CHECK(ncclGroupEnd());
  });
  return world_comm;
}

at::Tensor sparse_embedding_cuda_forward_reduce_scatter(
    // [B][T][D]
    at::Tensor embeddings) {
  // -> [B // #device][T][D]
  using namespace torch::cuda::nccl::detail;
  const int64_t world_size = sparse_embedding_comm().second;
  if (world_size == 1) {
    return embeddings;
  }

  at::cuda::CUDAGuard device_guard(embeddings.get_device());
  const auto B = embeddings.sizes()[0];
  const auto T = embeddings.sizes()[1];
  const auto D = embeddings.sizes()[2];
  auto output = at::empty({B / world_size, T, D}, embeddings.options());

  ncclDataType_t data_type = get_data_type(embeddings);
  int64_t count = embeddings.numel() / world_size;
  auto comm = nccl_comm();

  check_inputs({embeddings}, {output}, 1, world_size);
  auto stream =
      at::cuda::getCurrentCUDAStream(embeddings.get_device()).stream();

  {
    pybind11::gil_scoped_release no_gil;
    AutoNcclGroup nccl_group_guard;
    NCCL_CHECK(ncclReduceScatter(embeddings.data_ptr(), output.data_ptr(),
                                 count, data_type, ncclSum, comm, stream));
  }
  return output;
}

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}


at::Tensor sparse_embedding_cuda_forward_all2all_nccl(
  // [B][T // devices][D])
  at::Tensor embeddings) {

  // input: [B][T // devices][D]
  // reinterpret_input: [devices][B // devices][T // devices][D]
  // output: [B // devices][T][D]

  // at step w:
  // send input[w][B // devices][T // devices][D] to rank $w$
  // recv input[w][B // devices][T // devices][D] from rank $w$
  
  // now, after all-to-all, 
  // output is size [devices][B // devices][T // devices][D]
  // now, transpose to [B // devices][devices][T // devices][D]
  // now, view as [B // devices][devices * T // devices][D]
  // then, make contiguous.


  using namespace torch::cuda::nccl::detail;
  const int64_t world_size = sparse_embedding_comm().second;
  if (world_size == 1) {
    return embeddings;
  }

  const auto B = embeddings.size(0);
  const auto T = embeddings.size(1) * world_size;
  const auto D = embeddings.size(2);
  at::cuda::CUDAGuard device_guard(embeddings.get_device());
  AT_ASSERT(B % world_size == 0);

  auto all_to_all_output = at::empty({world_size, B / world_size, T / world_size, D}, embeddings.options());

  AT_ASSERT(embeddings.is_contiguous());
  AT_ASSERT(all_to_all_output.is_contiguous());

  AT_ASSERT(embeddings.numel() == all_to_all_output.numel());
  AT_ASSERT(embeddings.scalar_type() == all_to_all_output.scalar_type());


  ncclDataType_t data_type = get_data_type(embeddings);
  int64_t count = embeddings.numel() / world_size;
  const auto rank_offset = count * ncclTypeSize(data_type);
  auto comm = nccl_comm();

  AT_ASSERT(count == B * T * D / world_size / world_size);
  check_inputs({embeddings}, {all_to_all_output}, 1, 1);
  auto stream =
      at::cuda::getCurrentCUDAStream(embeddings.get_device()).stream();

  {
    pybind11::gil_scoped_release no_gil;
    AutoNcclGroup nccl_group_guard;
    for (int r = 0; r < world_size; r++) {
      // send all tables $t$ from rank $i$ to global batch chunk $j$.
      // recieve all tables $t$ from rank $j$ for global batch chunk $i$.
      NCCL_CHECK(ncclSend(((uint8_t*)embeddings.data_ptr()) + r * rank_offset, count, data_type, r, comm, stream));
      NCCL_CHECK(ncclRecv(((uint8_t*)all_to_all_output.data_ptr()) + r * rank_offset, count, data_type, r, comm, stream));
    }
  }
  auto transposed = all_to_all_output.transpose(1, 0);
  return transposed.contiguous().view({B / world_size, T, D});
}

at::Tensor sparse_embedding_cuda_forward_all_gather(
    // [B // #device][T][D]
    at::Tensor embeddings) {
  // -> [B][T][D]
  using namespace torch::cuda::nccl::detail;
  const int64_t world_size = sparse_embedding_comm().second;
  if (world_size == 1) {
    return embeddings;
  }
  at::cuda::CUDAGuard device_guard(embeddings.get_device());
  const auto B_div_world_size = embeddings.sizes()[0];
  const auto T = embeddings.sizes()[1];
  const auto D = embeddings.sizes()[2];
  auto output =
      at::empty({B_div_world_size * world_size, T, D}, embeddings.options());
  embeddings = embeddings.contiguous();

  ncclDataType_t data_type = get_data_type(embeddings);
  int64_t count = embeddings.numel();
  auto comm = nccl_comm();
  check_inputs({embeddings}, {output}, world_size, 1);
  auto stream =
      at::cuda::getCurrentCUDAStream(embeddings.get_device()).stream();

  {
    pybind11::gil_scoped_release no_gil;
    AutoNcclGroup nccl_group_guard;
    NCCL_CHECK(ncclAllGather(embeddings.data_ptr(), output.data_ptr(), count,
                             data_type, comm, stream));
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_embedding_cuda_forward,
        "sparse_embedding_cuda_"
        "forward(sharded_weights, "
        "scattered_indices) (CUDA)");
  m.def("forward_single", &sparse_embedding_cuda_forward_single,
        "sparse_embedding_cuda_forward_single(weights, indices) (CUDA)");
  m.def("forward_fast_single", &sparse_embedding_cuda_forward_fast_single,
        "sparse_embedding_cuda_forward_fast_single(weights, indices) (CUDA)");

  m.def("forward_offsets", &sparse_embedding_cuda_forward_offsets,
        "sparse_embedding_cuda_forward_offsets(weights, indices, offsets) (CUDA)");
  m.def("backward_update_offsets", &sparse_embedding_cuda_backward_update_offsets,
        "sparse_embedding_cuda_backward_update_offsets(weights, indices, offsets) (CUDA)");

  m.def("backward_update_single", &sparse_embedding_cuda_backward_update_single,
        "sparse_embedding_cuda_backward_update_single(grad_output, weights, "
        "indices, lr) (CUDA)");
  m.def("backward_update_fast_single",
        &sparse_embedding_cuda_backward_update_fast_single,
        "sparse_embedding_cuda_backward_update_fast_single(grad_output, "
        "weights, indices, lr) (CUDA)");

  m.def("forward_all2all", &sparse_embedding_cuda_forward_all2all,
        "sparse_embedding_cuda_forward_all2all(embeddings, result) (CUDA)");
  m.def("forward_all2all_nccl", &sparse_embedding_cuda_forward_all2all_nccl,
        "sparse_embedding_cuda_forward_all2all_nccl(embeddings, result) (CUDA)");


  m.def("forward_reducescatter", &sparse_embedding_cuda_forward_reduce_scatter,
        "sparse_embedding_cuda_forward_reduce_scatter(embeddings) (CUDA)");
  m.def("forward_allgather", &sparse_embedding_cuda_forward_all_gather,
        "sparse_embedding_cuda_forward_all_gather(embeddings) (CUDA)");
}
