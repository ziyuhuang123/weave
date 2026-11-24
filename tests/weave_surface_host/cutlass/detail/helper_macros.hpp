// Host-only stand-in for a CUDA / PyTorch / CUTLASS header that is absent from a
// CPU-only checkout (third-party/cutlass and third-party/deep_jit are empty
// submodules).  It provides ONLY the macros / types that the host-side Weave
// knob parsing chain references; no CUDA API is ever called by the test.
// See tests/test_21_weave_knobs_surface.py for the include-order contract.
#pragma once

#ifndef CUTLASS_HOST_DEVICE
#define CUTLASS_HOST_DEVICE
#endif
#ifndef CUTLASS_DEVICE
#define CUTLASS_DEVICE
#endif
#ifndef CUTLASS_HOST
#define CUTLASS_HOST
#endif
