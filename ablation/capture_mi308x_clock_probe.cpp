// Measure the exact offset between the kernel's s_memrealtime counter (the FI
// recorder's time base) and the host HSA system clock / CLOCK_MONOTONIC (the
// PMC sampler's time base), plus the true wall-clock rate.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void mr_kernel(unsigned long long* p)
{
    *p = __builtin_amdgcn_s_memrealtime();
}

extern "C" int mr_read(unsigned long long* out_ticks, int* wall_khz)
{
    unsigned long long* d = nullptr;
    if(hipMalloc(&d, 8) != hipSuccess) return -1;
    mr_kernel<<<1, 1>>>(d);
    if(hipDeviceSynchronize() != hipSuccess) { hipFree(d); return -2; }
    if(hipMemcpy(out_ticks, d, 8, hipMemcpyDeviceToHost) != hipSuccess) { hipFree(d); return -3; }
    hipFree(d);
    int k = 0;
    hipDeviceGetAttribute(&k, hipDeviceAttributeWallClockRate, 0);
    if(wall_khz) *wall_khz = k;
    return 0;
}
