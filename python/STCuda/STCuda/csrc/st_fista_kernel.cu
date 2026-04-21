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

template <typename scalar_t, typename ta_t>
__global__ void fista_gradient(
    const ta_t r_k,
    const ta_t s_k,
    const ta_t b,
    const scalar_t lam1,
    const ta_t lb,
    const ta_t ub,
    ta_t grad_a
  )
{
    // Each thread processes one element of the matrix
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    //const int X = gridDim.x * blockDim.x;
    const int Y = gridDim.y * blockDim.y;

    scalar_t d2x;

    // Map differential variables to pixel space description
    // of a 2D parameter space.

    if (y < Y-1)
        d2x = r_k[y][x];
    else
        d2x = grad_a[y][x];

    if(y)
        d2x -= r_k[y-1][x];

    if(x)
        d2x -= s_k[y][x-1];

    d2x += s_k[y][x];

    scalar_t a = b[y][x] - lam1 * d2x;

    //a = std::min(std::max(a, lb[y][x]), ub[y][x]);

    grad_a[y][x] = a;
}

template <typename scalar_t, typename ta_t>
__global__ void fista_iteration(
    const ta_t b,
    const scalar_t inv_8_lam1,
    const ta_t lb,
    const ta_t ub,
    const ta_t r_k,
    const ta_t s_k,
    const ta_t p_k,
    const ta_t q_k,
    ta_t r_kp1,
    ta_t s_kp1,
    ta_t p_kp1,
    ta_t q_kp1,
    ta_t grad_a,
    scalar_t t_k,
    scalar_t inv_t_kp1
  )
{

    // Each thread processes one element of the matrix
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int X = gridDim.x * blockDim.x;
    const int Y = gridDim.y * blockDim.y;

    constexpr scalar_t _1 = 1;

    bool m_ok = (y < Y-1);
    bool n_ok = (x < X-1);

    // Map the gradient to differential space
    scalar_t grad_ad_p, grad_ad_q;
    grad_ad_p = grad_ad_q = grad_a[y][x];
    if (n_ok)
        grad_ad_q -= grad_a[y][x+1];
    if (m_ok)
        grad_ad_p -= grad_a[y+1][x];

    // Update estimates of differential variables
    scalar_t p = r_k[y][x] + inv_8_lam1 * grad_ad_p;

    // Perform the projection step
    p = p / std::max(std::abs(p), _1);

    // Update normalized differential variables
    scalar_t r = p + (t_k - _1) * inv_t_kp1 * (p - p_k[y][x]);

    if (m_ok)
    {
        p_kp1[y][x] = p;
        r_kp1[y][x] = r;
    }

    // Update estimates of differential variables
    scalar_t q = s_k[y][x] + inv_8_lam1 * grad_ad_q;

    // Perform the projection step
    q = q / std::max(std::abs(q), _1);

    // Update normalized differential variables
    scalar_t s = q + (t_k - _1) * inv_t_kp1 * (q - q_k[y][x]);

    if (n_ok)
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

    int Y = b.size(0);
    int X = b.size(1);
    int y_div = 1;
    int x_div = 1;
    dim3 blocks(x_div, Y / y_div);
    dim3 threads(X / x_div, y_div);

    auto p_ping = at::zeros_like(b);
    auto p_pong = at::zeros_like(b);
    auto q_ping = at::zeros_like(b);
    auto q_pong = at::zeros_like(b);
    auto r_ping = at::zeros_like(b);
    auto r_pong = at::zeros_like(b);
    auto s_ping = at::zeros_like(b);
    auto s_pong = at::zeros_like(b);
    auto grad_a = at::zeros_like(b);

    AT_DISPATCH_FLOATING_TYPES(b.scalar_type(), "fista_launch",
    ([&]{
        using scalar_t = scalar_t;

        auto ta_b      =      b.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_lb     =     lb.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_ub     =     ub.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_grad_a = grad_a.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_p_ping = p_ping.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_p_pong = p_pong.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_q_ping = q_ping.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_q_pong = q_pong.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_r_ping = r_ping.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_r_pong = r_pong.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_s_ping = s_ping.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();
        auto ta_s_pong = s_pong.packed_accessor64<scalar_t, 2, at::RestrictPtrTraits>();

        scalar_t t_k = 1;
        scalar_t lam1_ = lam1.item<scalar_t>();
        scalar_t inv_8_lam1 = scalar_t(1) / ( scalar_t(8) * lam1_);

        auto stream = at::cuda::getCurrentCUDAStream();

        for (int iter = 0; iter < num_iter; ++iter)
        {

            scalar_t t_kp1 = (scalar_t(1) + std::sqrt(scalar_t(1) + scalar_t(4) * t_k * t_k) ) / scalar_t(2);
            scalar_t inv_t_kp1 = scalar_t(1) / t_kp1;

            if ( iter % 2 == 0 )
            {
                // Compute from scratch buffer ping into pong

                fista_gradient<scalar_t><<<blocks, threads, 0, stream>>>(
                    ta_r_ping, ta_s_ping,
                    ta_b, lam1_, ta_lb, ta_ub, ta_grad_a );

                if ( iter < num_iter-1 )
                {
                    fista_iteration<scalar_t><<<blocks, threads, 0, stream>>>(
                        ta_b, inv_8_lam1, ta_lb, ta_ub,
                        ta_r_ping, ta_s_ping, ta_p_ping, ta_q_ping,
                        ta_r_pong, ta_s_pong, ta_p_pong, ta_q_pong,
                        ta_grad_a, t_k, inv_t_kp1 );
                }
            }
            else
            {
                // Compute from scratch buffer pong into ping

                fista_gradient<scalar_t><<<blocks, threads, 0, stream>>>(
                    ta_r_pong, ta_s_pong,
                    ta_b, lam1_, ta_lb, ta_ub, ta_grad_a );

                if ( iter < num_iter-1 )
                {
                    fista_iteration<scalar_t><<<blocks, threads, 0, stream>>>(
                        ta_b, inv_8_lam1, ta_lb, ta_ub,
                        ta_r_pong, ta_s_pong, ta_p_pong, ta_q_pong,
                        ta_r_ping, ta_s_ping, ta_p_ping, ta_q_ping,
                        ta_grad_a, t_k, inv_t_kp1 );
                }
            }

            t_k = t_kp1;
        }
    }));

    return grad_a;
}

} // namespace STCuda

