// st_fista_kernel.h
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2023-2026 University Corporation for Atmospheric Research
// All rights reserved.

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace STCuda
{
    at::Tensor fista_launch(
        const at::Tensor& b,
        const at::Tensor& lam1,
        const at::Tensor& lb,
        const at::Tensor& ub);

}

