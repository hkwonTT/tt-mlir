// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn-precompiled.hpp"
::ttnn::Tensor mesh_shards(::ttnn::Tensor v1) {
  ttnn::distributed::MeshDevice* v2 = ttnn::DeviceGetter::getInstance();
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v3 = ttnn::mesh_shard(v1);
  ttnn::deallocate(v1, false);
  ::ttnn::Tensor v4 = ttnn::to_layout(v3, ::ttnn::Layout::TILE, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v3, false);
  ::ttnn::Tensor v5 = ttnn::to_device(v4, v2, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::cos(v5, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});
  ttnn::deallocate(v5, false);
  ::ttnn::Tensor v7 = ttnn::from_device(v6);
  ttnn::deallocate(v6, false);
  ::ttnn::Tensor v8 = ttnn::to_layout(v7, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY}, static_cast<::ttnn::distributed::MeshDevice *>(nullptr));
  ttnn::deallocate(v7, false);
  assert(0 && "Mesh shard operation is not supported in emitc yet."); // ::ttnn::mesh_shard
  ::ttnn::Tensor v9 = ttnn::mesh_shard(v8);
  ttnn::deallocate(v8, false);
  return v9;
}

::ttnn::Tensor create_inputs_for_mesh_shards() {
  ::ttnn::Tensor v1 = ttnn::ones(::ttnn::Shape({256, 256}), ::ttnn::DataType::FLOAT32, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::SYSTEM_MEMORY});
  return v1;
}

int32_t main() {
  ::ttnn::Tensor v1 = create_inputs_for_mesh_shards();
  ::ttnn::Tensor v2 = mesh_shards(v1);
  int32_t v3 = 0;
  return v3;
}
