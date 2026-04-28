// st_lidar_fwd_kernel.cu
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2026 University Corporation for Atmospheric Research
// All rights reserved.

#include "st_lidar_fwd_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace STCuda
{

template <typename scalar_t, int DIMS>
using accessor_t = at::PackedTensorAccessor32<scalar_t, DIMS, at::RestrictPtrTraits>;

// One block per (t, r) pixel; BLOCK_F threads cooperate over the frequency dimension.
// BLOCK_F is FREQ padded up to a power of two in order to do the sum-reduce as repeated folds.
// The padding threads write zeros. Shared memory size is four scalar_t[BLOCK_F].

template <typename scalar_t, int BLOCK_F>
__global__ void lidar_fwd_kernel(
    const accessor_t<scalar_t, 3> od,
    const accessor_t<scalar_t, 3> bs,
    const accessor_t<scalar_t, 3> spec,
    const accessor_t<scalar_t, 1> rx,
    accessor_t<scalar_t, 2>       out,
    int T, int R, int FREQ
)
{
    __shared__ scalar_t e_sh[BLOCK_F];
    __shared__ scalar_t a_sh[BLOCK_F];
    __shared__ scalar_t bs_sh[BLOCK_F];
    __shared__ scalar_t sum_sh[BLOCK_F];

    const int f = threadIdx.x;
    const int r = blockIdx.x;
    const int t = blockIdx.y;

    if (r >= R || t >= T) return;

    if (f < FREQ) {
        scalar_t e_f = exp(-od[t][r][f]);
        e_sh[f]  = e_f;
        a_sh[f]  = e_f * spec[t][0][f];
        bs_sh[f] = bs[t][r][f];
    }
    __syncthreads();

    scalar_t partial = scalar_t(0);
    if (f < FREQ) {
        const int PAD = (FREQ - 1) / 2;
        const int k_lo = (f < PAD) ? (PAD - f) : 0;
        const int k_hi = (f + PAD < FREQ) ? FREQ : (FREQ + PAD - f);

        scalar_t conv_val = scalar_t(0);
        for (int k = k_lo; k < k_hi; k++)
            conv_val += a_sh[f + k - PAD] * bs_sh[k];

        partial = conv_val * rx[f] * e_sh[f];
    }

    sum_sh[f] = partial;
    __syncthreads();

    // sum by repeated fold
    for (int s = BLOCK_F / 2; s >= 1; s >>= 1) {
        if (f < s) sum_sh[f] += sum_sh[f + s];
        __syncthreads();
    }

    if (f == 0) out[t][r] = sum_sh[0];
}

at::Tensor lidar_fwd_launch(
    const at::Tensor& od,
    const at::Tensor& bs,
    const at::Tensor& spec,
    const at::Tensor& rx
)
{
    const int T = od.size(0);
    const int R = od.size(1);
    const int F = od.size(2);

    TORCH_CHECK(F <= 256,
        "lidar_fwd_launch: F=", F, " exceeds max supported (256)");
    TORCH_CHECK(bs.sizes() == od.sizes(),
        "lidar_fwd_launch: bs/od shape mismatch");
    TORCH_CHECK(spec.size(0) == T && spec.size(1) == 1 && spec.size(2) == F,
        "lidar_fwd_launch: spec must be (T, 1, F)");
    TORCH_CHECK(rx.size(0) == F,
        "lidar_fwd_launch: rx must be (F,)");

    auto out    = at::empty({T, R}, od.options());
    auto stream = at::cuda::getCurrentCUDAStream();

    dim3 blocks(R, T);

    AT_DISPATCH_FLOATING_TYPES(od.scalar_type(), "lidar_fwd_launch", ([&] {
        auto ta_od   =   od.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
        auto ta_bs   =   bs.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
        auto ta_spec = spec.packed_accessor32<scalar_t, 3, at::RestrictPtrTraits>();
        auto ta_rx   =   rx.packed_accessor32<scalar_t, 1, at::RestrictPtrTraits>();
        auto ta_out  =  out.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

        if (F <= 64) {
            lidar_fwd_kernel<scalar_t, 64><<<blocks, 64, 0, stream>>>(
                ta_od, ta_bs, ta_spec, ta_rx, ta_out, T, R, F);
        } else if (F <= 128) {
            lidar_fwd_kernel<scalar_t, 128><<<blocks, 128, 0, stream>>>(
                ta_od, ta_bs, ta_spec, ta_rx, ta_out, T, R, F);
        } else {
            lidar_fwd_kernel<scalar_t, 256><<<blocks, 256, 0, stream>>>(
                ta_od, ta_bs, ta_spec, ta_rx, ta_out, T, R, F);
        }
    }));

    return out;
}

} // namespace STCuda
