#pragma once

template <typename real>
struct coefficients {
    real diffusion_rate_u = 0.1;
    real diffusion_rate_v = 0.05;
    real feed_rate = 0.014;
    real kill_rate = 0.054;
    real dt = 1.0;
};
    