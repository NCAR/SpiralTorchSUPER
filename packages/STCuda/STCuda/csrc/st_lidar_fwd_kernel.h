// st_lidar_fwd_kernel.h
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2026 University Corporation for Atmospheric Research
// All rights reserved.

#pragma once

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

namespace STCuda
{
    // Fused lidar forward model: spectral convolution + integration.
    // Returns rho0 (T, R) = sum_freq[ conv(exp(-od)*spec, bs) * rx * exp(-od) ]
    // Caller should multiply by remaining calibration scalars (r2_loss, mol_bs, common, etc.)
    at::Tensor lidar_fwd_launch(
        const at::Tensor& od,    // (T, R, F)  optical depth
        const at::Tensor& bs,    // (T, R, F)  backscatter spectrum
        const at::Tensor& spec,  // (T, 1, F)  laser wavelength histogram
        const at::Tensor& rx     // (F,)       receiver transmission
    );

} // namespace STCuda
