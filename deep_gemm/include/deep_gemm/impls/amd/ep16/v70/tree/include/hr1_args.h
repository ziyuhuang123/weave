#pragma once
// POD argument block shared by the device kernels and the host launcher.
// Kept free of any HIP device-only constructs so the host pass can include it.
#include <stdint.h>

#ifndef HR1_ARGS_DEFINED
#define HR1_ARGS_DEFINED
struct Hr1Args {
    const int32_t* ids;
    const uint8_t* a;
    const uint8_t* b;
    const float* a_scale;
    const float* b_scale;
    uint8_t* q_out;
    float* q_scale;
    const float* weights;
    uint16_t* c_out;
    unsigned row_tiles, n_blocks, tiles_per_expert;
    unsigned valid_rows, num_tokens, route_rows, N, K;
    unsigned total_tiles, ntile_major;
};
#endif
