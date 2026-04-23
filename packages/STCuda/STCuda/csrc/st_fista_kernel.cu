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

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int NUM_ITER = 50;

template <typename scalar_t, typename accessor_t>
__global__ void fista_iterate(
    accessor_t        r,
    accessor_t        s,
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

    const bool valid     = (x < width) && (y < height);
    const bool has_right = (x < width  - 1);
    const bool has_below = (y < height - 1);

    scalar_t t_k = one;

    // Register our own cell: r_yx means r[y][x]
    // For r, s, b, grad_a the regs cache the tensors
    // For p, q there is no tensor, use your imagination
    const scalar_t b_yx      = valid ? b[y][x] : scalar_t(0);
    scalar_t       r_yx      = scalar_t(0);
    scalar_t       s_yx      = scalar_t(0);
    scalar_t       p_yx      = scalar_t(0);
    scalar_t       q_yx      = scalar_t(0);
    scalar_t       grad_a_yx = scalar_t(0);

    for (int iter = 0; iter < NUM_ITER; ++iter)
    {
        const scalar_t t_kp1     = (one + sqrt(one + scalar_t(4) * t_k * t_k)) / scalar_t(2);
        const scalar_t inv_t_kp1 = one / t_kp1;
        const bool     last_iter = (iter == NUM_ITER - 1);

        // -----------------------
        // Phase 1: compute grad_a
        // -----------------------
        if (valid)
        {
            scalar_t d2x;

            if (has_below)
                d2x = r_yx;
            else
                d2x = grad_a_yx;

            if (y)
                d2x -= r[y-1][x];

            if (x)
                d2x -= s[y][x-1];

            d2x += s_yx;

            grad_a_yx = b_yx - lam1 * d2x;

            //a = std::min(std::max(a, lb[y][x]), ub[y][x]);

            grad_a[y][x] = grad_a_yx;
        }

        // Wait for completed grad_a
        grid.sync();

        // -----------------------
        // Phase 2: iteration step
        // -----------------------
        if (!last_iter && valid && has_below)
        {
            // Map the gradient to differential space
            scalar_t grad_ad_p = grad_a_yx - grad_a[y+1][x];

            // p update + projection
            scalar_t p_new = r_yx + inv_8_lam1 * grad_ad_p;
            p_new = p_new / max(abs(p_new), one);

            // Update normalized differential variables
            scalar_t r_new = p_new + (t_k - one) * inv_t_kp1 * (p_new - p_yx);

            p_yx = p_new;
            r[y][x] = r_yx = r_new;
        }

        if (!last_iter && valid && has_right)
        {
            // Map the gradient to differential space
            scalar_t grad_ad_q = grad_a_yx - grad_a[y][x+1];

            // q update + projection
            scalar_t q_new = s_yx + inv_8_lam1 * grad_ad_q;
            q_new = q_new / max(abs(q_new), one);

            // Update normalized differential variables
            scalar_t s_new = q_new + (t_k - one) * inv_t_kp1 * (q_new - q_yx);

            q_yx = q_new;
            s[y][x] = s_yx = s_new;
        }

        // Wait for completed r/s
        grid.sync();

        t_k = t_kp1;
    }
}

// Fallback

template <typename scalar_t, typename accessor_t>
__global__ void fista_gradient(
    const accessor_t  r_k,
    const accessor_t  s_k,
    const accessor_t  b,
    scalar_t          lam1,
    accessor_t        grad_a,
    int               width,
    int               height
)
{
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    if ((x >= width) || (y >= height))
        return;

    constexpr scalar_t one = scalar_t(1);

    const bool has_right = (x < width  - 1);
    const bool has_below = (y < height - 1);

    scalar_t d2x;

    if (has_below)
        d2x = r_k[y][x];
    else
        d2x = grad_a[y][x];

    if (y)
        d2x -= r_k[y-1][x];

    if (x)
        d2x -= s_k[y][x-1];

    d2x += s_k[y][x];

    grad_a[y][x] = b[y][x] - lam1 * d2x;

    //a = std::min(std::max(a, lb[y][x]), ub[y][x]);
}

template <typename scalar_t, typename accessor_t>
__global__ void fista_iteration(
    const accessor_t  r_k,
    const accessor_t  s_k,
    const accessor_t  p_k,
    const accessor_t  q_k,
    accessor_t        r_kp1,
    accessor_t        s_kp1,
    accessor_t        p_kp1,
    accessor_t        q_kp1,
    const accessor_t  grad_a,
    scalar_t          inv_8_lam1,
    scalar_t          t_k,
    scalar_t          inv_t_kp1,
    int               width,
    int               height
)
{
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;

    if ((x >= width) || (y >= height))
        return;

    constexpr scalar_t one = scalar_t(1);

    const bool has_right = (x < width  - 1);
    const bool has_below = (y < height - 1);

    if (has_below)
    {
        scalar_t grad_ad_p = grad_a[y][x] - grad_a[y+1][x];

        // p update + projection
        scalar_t p_new = r_k[y][x] + inv_8_lam1 * grad_ad_p;
        p_new = p_new / max(abs(p_new), one);

        // Update normalized differential variables
        p_kp1[y][x] = p_new;
        r_kp1[y][x] = p_new + (t_k - one) * inv_t_kp1 * (p_new - p_k[y][x]);
    }

    if (has_right)
    {
        scalar_t grad_ad_q = grad_a[y][x] - grad_a[y][x+1];

        // q update + projection
        scalar_t q_new = s_k[y][x] + inv_8_lam1 * grad_ad_q;
        q_new = q_new / max(abs(q_new), one);

        // Update normalized differential variables
        q_kp1[y][x] = q_new;
        s_kp1[y][x] = q_new + (t_k - one) * inv_t_kp1 * (q_new - q_k[y][x]);
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

    auto grad_a = at::zeros_like(b);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "fista_launch", ([&] {
        using scalar_t = scalar_t;

        auto ta_b      =      b.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_lb     =     lb.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_ub     =     ub.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_grad_a = grad_a.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

        scalar_t lam1_      = lam1.item<scalar_t>();
        scalar_t inv_8_lam1 = scalar_t(1) / (scalar_t(8) * lam1_);

        using kernel_t = decltype(fista_iterate<scalar_t, decltype(ta_b)>);
        kernel_t* kernel = fista_iterate<scalar_t, decltype(ta_b)>;

        int max_blocks_per_sm, sm_count;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, BLOCK_X * BLOCK_Y, 0);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, at::cuda::current_device());

        if (int(blocks.x * blocks.y) <= max_blocks_per_sm * sm_count)
        {
            auto r = at::zeros_like(b);
            auto s = at::zeros_like(b);

            auto ta_r = r.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_s = s.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

            void* args[] = {
                (void*)&ta_r,
                (void*)&ta_s,
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
        }
        else
        {
            // Fallback: The grid is larger than the device, use separate kernels for synchronization
            auto r_ping = at::zeros_like(b);
            auto r_pong = at::zeros_like(b);
            auto s_ping = at::zeros_like(b);
            auto s_pong = at::zeros_like(b);
            auto p_ping = at::zeros_like(b);
            auto p_pong = at::zeros_like(b);
            auto q_ping = at::zeros_like(b);
            auto q_pong = at::zeros_like(b);

            auto ta_r_ping = r_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_r_pong = r_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_s_ping = s_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_s_pong = s_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_p_ping = p_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_p_pong = p_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_q_ping = q_ping.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();
            auto ta_q_pong = q_pong.packed_accessor32<scalar_t, 2, at::RestrictPtrTraits>();

            scalar_t t_k = scalar_t(1);

            for (int iter = 0; iter < NUM_ITER; ++iter)
            {
                const scalar_t t_kp1     = (scalar_t(1) + sqrt(scalar_t(1) + scalar_t(4) * t_k * t_k)) / scalar_t(2);
                const scalar_t inv_t_kp1 = scalar_t(1) / t_kp1;
                const bool     ping      = (iter % 2 == 0);

                auto& ta_r_k   = ping ? ta_r_ping : ta_r_pong;
                auto& ta_s_k   = ping ? ta_s_ping : ta_s_pong;
                auto& ta_p_k   = ping ? ta_p_ping : ta_p_pong;
                auto& ta_q_k   = ping ? ta_q_ping : ta_q_pong;
                auto& ta_r_kp1 = ping ? ta_r_pong : ta_r_ping;
                auto& ta_s_kp1 = ping ? ta_s_pong : ta_s_ping;
                auto& ta_p_kp1 = ping ? ta_p_pong : ta_p_ping;
                auto& ta_q_kp1 = ping ? ta_q_pong : ta_q_ping;

                fista_gradient<scalar_t><<<blocks, threads, 0, stream>>>(
                    ta_r_k, ta_s_k,
                    ta_b, lam1_,
                    ta_grad_a,
                    width, height);

                if (iter < NUM_ITER - 1)
                {
                    fista_iteration<scalar_t><<<blocks, threads, 0, stream>>>(
                        ta_r_k,   ta_s_k,   ta_p_k,   ta_q_k,
                        ta_r_kp1, ta_s_kp1, ta_p_kp1, ta_q_kp1,
                        ta_grad_a, inv_8_lam1, t_k, inv_t_kp1,
                        width, height);
                }

                t_k = t_kp1;
            }
        }
    }));

    return grad_a;
}

} // namespace STCuda
