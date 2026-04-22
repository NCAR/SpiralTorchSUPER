// st_fista_kernel.cu
//
// Author: Adam Karboski <karboski@ucar.edu>
//
// Copyright © 2023-2026 University Corporation for Atmospheric Research
// All rights reserved.

#include "st_fista_kernel.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

namespace STCuda
{

// Thread block size
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

template <typename scalar_t, typename accessor_t>
__global__ void fista_iteration_fused(
    const accessor_t  r_k,
    const accessor_t  s_k,
    const accessor_t  b,
    scalar_t          lam1,
    const accessor_t  lb,
    const accessor_t  ub,
    const accessor_t  p_k,
    const accessor_t  q_k,
    accessor_t        r_kp1,
    accessor_t        s_kp1,
    accessor_t        p_kp1,
    accessor_t        q_kp1,
    accessor_t        grad_a,
    scalar_t          t_k,
    scalar_t          inv_t_kp1,
    scalar_t          inv_8_lam1,
    int               width,
    int               height,
    bool              last_iter
)
{
    // Index within the image
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    const int thread_x = threadIdx.x;
    const int thread_y = threadIdx.y;

    // Shared memory tile for grad_a.
    // Interior: BLOCK_Y rows x BLOCK_X cols  (one value per thread)
    // Halo:     one extra col on the right, one extra row on the bottom
    // Layout:   row-major, stride = BLOCK_X+1
    __shared__ scalar_t smem_grad[(BLOCK_Y + 1) * (BLOCK_X + 1)];

    const int smem_idx = thread_y * (BLOCK_X + 1) + thread_x;

    constexpr scalar_t one = scalar_t(1);

    const bool valid = (x < width) && (y < height);

    // -----------------------
    // Phase 1: compute grad_a
    // -----------------------
    scalar_t ga = scalar_t(0);

    if (valid)
    {
        scalar_t d2x;

        if (y < height - 1)
            d2x = r_k[y][x];
        else
            d2x = grad_a[y][x];

        if (y)
            d2x -= r_k[y-1][x];

        if (x)
            d2x -= s_k[y][x-1];

        d2x += s_k[y][x];

        ga = b[y][x] - lam1 * d2x;

        grad_a[y][x] = ga;
    }

    // Stage interior value into smem
    smem_grad[smem_idx] = valid ? ga : scalar_t(0);

    // Right halo: threads in the last column of the block load (x+1, y)
    if (thread_x == BLOCK_X - 1)
    {
        const int halo_x = x + 1;
        smem_grad[thread_y * (BLOCK_X + 1) + BLOCK_X] =
            (halo_x < width && valid) ? grad_a[y][halo_x] : scalar_t(0);
    }

    // Bottom halo: threads in the last row of the block load (x, y+1)
    if (thread_y == BLOCK_Y - 1)
    {
        const int halo_y = y + 1;
        smem_grad[BLOCK_Y * (BLOCK_X + 1) + thread_x] =
            (halo_y < height && valid) ? grad_a[halo_y][x] : scalar_t(0);
    }

    __syncthreads();

    // -----------------------
    // Phase 2: iteration step
    // -----------------------
    if (!valid || last_iter) return;

    const bool has_right = (x < width  - 1);
    const bool has_below = (y < height - 1);

    // Map grad_a to differential space; neighbours come from smem
    scalar_t grad_ad_p = ga;
    scalar_t grad_ad_q = ga;

    if (has_right)
        grad_ad_q -= smem_grad[thread_y * (BLOCK_X + 1) + (thread_x + 1)];   // (x+1, y)

    if (has_below)
        grad_ad_p -= smem_grad[(thread_y + 1) * (BLOCK_X + 1) + thread_x];   // (x, y+1)

    // p update + projection
    scalar_t p = r_k[y][x] + inv_8_lam1 * grad_ad_p;
    p = p / max(abs(p), one);

    // Momentum extrapolation
    scalar_t r = p + (t_k - one) * inv_t_kp1 * (p - p_k[y][x]);

    if (has_below)
    {
        p_kp1[y][x] = p;
        r_kp1[y][x] = r;
    }

    // q update + projection
    scalar_t q = s_k[y][x] + inv_8_lam1 * grad_ad_q;
    q = q / max(abs(q), one);

    scalar_t s = q + (t_k - one) * inv_t_kp1 * (q - q_k[y][x]);

    if (has_right)
    {
        q_kp1[y][x] = q;
        s_kp1[y][x] = s;
    }
}


at::Tensor fista_launch(
    const at::Tensor& b,
    const at::Tensor& lam1,
    const at::Tensor& lb,
    const at::Tensor& ub
)
{
    constexpr int num_iter = 50;

    const int height = b.size(0);
    const int width  = b.size(1);

    dim3 threads(BLOCK_X, BLOCK_Y);
    dim3 blocks((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    auto p_ping = at::zeros_like(b);
    auto p_pong = at::zeros_like(b);
    auto q_ping = at::zeros_like(b);
    auto q_pong = at::zeros_like(b);
    auto r_ping = at::zeros_like(b);
    auto r_pong = at::zeros_like(b);
    auto s_ping = at::zeros_like(b);
    auto s_pong = at::zeros_like(b);
    auto grad_a = at::zeros_like(b);

    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "fista_launch", ([&] {
        using scalar_t = scalar_t;

        auto ta_b      =      b.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_lb     =     lb.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_ub     =     ub.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_grad_a = grad_a.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_p_ping = p_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_p_pong = p_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_q_ping = q_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_q_pong = q_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_r_ping = r_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_r_pong = r_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_s_ping = s_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_s_pong = s_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

        scalar_t t_k        = scalar_t(1);
        scalar_t lam1_      = lam1.item<scalar_t>();
        scalar_t inv_8_lam1 = scalar_t(1) / (scalar_t(8) * lam1_);

        auto stream = at::cuda::getCurrentCUDAStream();

        for (int iter = 0; iter < num_iter; ++iter)
        {
            const scalar_t t_kp1     = (scalar_t(1) + std::sqrt(scalar_t(1) + scalar_t(4) * t_k * t_k)) / scalar_t(2);
            const scalar_t inv_t_kp1 = scalar_t(1) / t_kp1;
            const bool     last_iter = (iter == num_iter - 1);
            const bool     ping      = (iter % 2 == 0);

            fista_iteration_fused<scalar_t><<<blocks, threads, 0, stream>>>
            (
                ping ? ta_r_ping : ta_r_pong,
                ping ? ta_s_ping : ta_s_pong,
                ta_b,
                lam1_,
                ta_lb,
                ta_ub,
                ping ? ta_p_ping : ta_p_pong,
                ping ? ta_q_ping : ta_q_pong,
                ping ? ta_r_pong : ta_r_ping,
                ping ? ta_s_pong : ta_s_ping,
                ping ? ta_p_pong : ta_p_ping,
                ping ? ta_q_pong : ta_q_ping,
                ta_grad_a,
                t_k,
                inv_t_kp1,
                inv_8_lam1,
                width,
                height,
                last_iter
            );

            t_k = t_kp1;
        }
    }));

    return grad_a;
}

} // namespace STCuda
