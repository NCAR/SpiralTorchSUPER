// st_lidar_fwd.cpp
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2026 University Corporation for Atmospheric Research
// All rights reserved.

#include "st_lidar_fwd_kernel.h"

namespace STCuda
{

at::Tensor st_lidar_fwd(
    const at::Tensor& od,
    const at::Tensor& bs,
    const at::Tensor& spec,
    const at::Tensor& rx)
{
    TORCH_CHECK(od.is_cuda(),   "od must be on GPU");
    TORCH_CHECK(bs.is_cuda(),   "bs must be on GPU");
    TORCH_CHECK(spec.is_cuda(), "spec must be on GPU");
    TORCH_CHECK(rx.is_cuda(),   "rx must be on GPU");

    return lidar_fwd_launch(od, bs, spec, rx);
}

TORCH_LIBRARY_FRAGMENT(STCuda, m) {
    m.def("st_lidar_fwd(Tensor od, Tensor bs, Tensor spec, Tensor rx) -> Tensor");
}

TORCH_LIBRARY_IMPL(STCuda, CUDA, m) {
    m.impl("st_lidar_fwd", &st_lidar_fwd);
}

} // namespace STCuda
