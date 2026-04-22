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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace STCuda
{

// Thread block size
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int NUM_ITER = 50;

template <typename scalar_t, typename accessor_t>
__global__ void fista_persistent(
    accessor_t   r_ping,
    accessor_t   s_ping,
    accessor_t   p_ping,
    accessor_t   q_ping,
    accessor_t   r_pong,
    accessor_t   s_pong,
    accessor_t   p_pong,
    accessor_t   q_pong,
    const accessor_t  b,
    scalar_t          lam1,
    const accessor_t  lb,
    const accessor_t  ub,
    accessor_t        grad_a,
    scalar_t          inv_8_lam1,
    int               width,
    int               height
)
{
    cg::grid_group grid = cg::this_grid();

    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    constexpr scalar_t one = scalar_t(1);

    const bool valid = (x < width) && (y < height);
    const bool has_right = valid && (x < width  - 1);
    const bool has_below = valid && (y < height - 1);

    scalar_t t_k = one;

    for (int iter = 0; iter < NUM_ITER; ++iter)
    {
        const scalar_t t_kp1     = (one + sqrt(one + scalar_t(4) * t_k * t_k)) / scalar_t(2);
        const scalar_t inv_t_kp1 = one / t_kp1;
        const bool     last_iter = (iter == NUM_ITER - 1);
        const bool     ping      = (iter % 2 == 0);

        // Select ping-pong buffers for this iteration
        accessor_t& r_k   = ping ? r_ping : r_pong;
        accessor_t& s_k   = ping ? s_ping : s_pong;
        accessor_t& p_k   = ping ? p_ping : p_pong;
        accessor_t& q_k   = ping ? q_ping : q_pong;
        accessor_t& r_kp1 = ping ? r_pong : r_ping;
        accessor_t& s_kp1 = ping ? s_pong : s_ping;
        accessor_t& p_kp1 = ping ? p_pong : p_ping;
        accessor_t& q_kp1 = ping ? q_pong : q_ping;

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

        // Wait for completed grad_a
        grid.sync();

        // -----------------------
        // Phase 2: iteration step
        // -----------------------
        if (!last_iter && valid)
        {
            const bool has_right = (x < width  - 1);
            const bool has_below = (y < height - 1);

            scalar_t grad_ad_p = ga;
            scalar_t grad_ad_q = ga;

            if (has_right)
                grad_ad_q -= grad_a[y][x+1];

            if (has_below)
                grad_ad_p -= grad_a[y+1][x];

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

        // Wait for completed p/q/r/s
        grid.sync();

        t_k = t_kp1;
    }
}


at::Tensor fista_launch(
    const at::Tensor& b,
    const at::Tensor& lam1,
    const at::Tensor& lb,
    const at::Tensor& ub
)
{
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

        scalar_t lam1_      = lam1.item<scalar_t>();
        scalar_t inv_8_lam1 = scalar_t(1) / (scalar_t(8) * lam1_);

        auto stream = at::cuda::getCurrentCUDAStream();

        using kernel_t = decltype(fista_persistent<scalar_t, decltype(ta_b)>);
        kernel_t* kernel = fista_persistent<scalar_t, decltype(ta_b)>;

        int max_blocks_per_sm, sm_count;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BLOCK_X * BLOCK_Y, 0);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, at::cuda::current_device());

        TORCH_CHECK( (int)(blocks.x * blocks.y) <= max_blocks_per_sm * sm_count,
                     "Grid too large for cooperative launch: ", blocks.x * blocks.y,
                     " blocks, max=", max_blocks_per_sm * sm_count);

        void* args[] = {
            (void*)&ta_r_ping,
            (void*)&ta_s_ping,
            (void*)&ta_p_ping,
            (void*)&ta_q_ping,
            (void*)&ta_r_pong,
            (void*)&ta_s_pong,
            (void*)&ta_p_pong,
            (void*)&ta_q_pong,
            (void*)&ta_b,
            (void*)&lam1_,
            (void*)&ta_lb,
            (void*)&ta_ub,
            (void*)&ta_grad_a,
            (void*)&inv_8_lam1,
            (void*)&width,
            (void*)&height
        };

        cudaLaunchCooperativeKernel((void*)kernel, blocks, threads, args, 0, stream);
    }));

    return grad_a;
}

} // namespace STCuda
