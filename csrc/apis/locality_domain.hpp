#pragma once

#include <torch/python.h>

#include "../runtime/runtime.hpp"
#include "../utils/exception.hpp"

namespace deep_gemm::locality_domain {

// SM103 (GB300-class) is a dual-die package: two locality domains, 74 SMs each
// on the 148-SM device. TPCs never straddle the die boundary.
static constexpr int kNumDeviceLocalityDomains = 2;
static constexpr int kNumSMsPerTPC = 2;

static int get_num_locality_domains() {
    return kNumDeviceLocalityDomains;
}

// A true SM->die probe needs allocations whose physical home is a locality
// domain. CUDA 13.4 exposes that natively; older drivers require MLOPart/MPS
// (the AKA path). Detect the latter without making it mandatory.
static bool is_localization_available() {
    static const bool available = []() {
        if (deep_jit::get_env<int>("AKA_ENABLE_MLOPART", 0) == 0)
            return false;
        try {
            pybind11::module_::import("aka_mlopart");
            return true;
        } catch (...) {
            return false;
        }
    }();
    return available;
}

// Canonical fallback used by upstream DeepGEMM when localization is unavailable:
// even TPCs belong to domain 0, odd TPCs to domain 1, and both SMs of a TPC
// share the domain. On 148 SMs this is exactly 37 TPCs / 74 SMs per die.
static torch::Tensor get_even_sm_locality_domains() {
    const auto num_sms = jit->device.get_num_sms();
    DG_HOST_ASSERT(num_sms % (kNumSMsPerTPC * kNumDeviceLocalityDomains) == 0);
    const auto tpc_idx = torch::arange(
        num_sms / kNumSMsPerTPC,
        torch::dtype(torch::kInt).device(torch::kCUDA));
    return (tpc_idx % kNumDeviceLocalityDomains)
        .to(torch::kUInt8)
        .repeat_interleave(kNumSMsPerTPC);
}

// If the probed map is off by a few TPCs, move the surplus TPCs of the larger
// domain to the smaller one so both counters see the same number of schedulers.
static torch::Tensor balance_sm_locality_domains(const torch::Tensor& sm_domain) {
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto tpc_domain = tpcs.select(1, 0);
    DG_HOST_ASSERT(tpcs.size(0) % 2 == 0);

    const int64_t num_tpcs = tpcs.size(0);
    const int64_t num_domain_1 = tpc_domain.sum().item<int64_t>();
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;
    // C++ slice(dim, start) defaults end to INT64_MAX: keep the tail surplus.
    const auto surplus = (tpc_domain == larger).nonzero().flatten().slice(0, num_tpcs / 2);
    const auto balanced = sm_domain.clone();
    balanced.view({-1, kNumSMsPerTPC}).index_fill_(0, surplus, larger ^ 1);
    return balanced;
}

static torch::Tensor get_sm_locality_domains() {
    if (not runtime->sm_locality_domains.defined()) {
        // The probe path is intentionally conservative: this host (driver
        // 580.105.08) has neither CUDA 13.4 locality-domain mem APIs nor
        // aka_mlopart, so use the upstream even mapping. The scheduler contract
        // is identical once a probed tensor is supplied.
        runtime->sm_locality_domains = get_even_sm_locality_domains();
    }
    return runtime->sm_locality_domains;
}

static torch::Tensor get_balanced_sm_locality_domains() {
    if (not runtime->balanced_sm_locality_domains.defined())
        runtime->balanced_sm_locality_domains = balance_sm_locality_domains(get_sm_locality_domains());
    return runtime->balanced_sm_locality_domains;
}

static void register_apis(pybind11::module_& m) {
    auto submodule = m.def_submodule("locality_domain", "SM/task affinity for GPU locality domains");
    submodule.def("get_num_locality_domains", &get_num_locality_domains);
    submodule.def("is_localization_available", &is_localization_available);
    submodule.def("get_sm_locality_domains", &get_sm_locality_domains);
    submodule.def("get_balanced_sm_locality_domains", &get_balanced_sm_locality_domains);
}

}  // namespace deep_gemm::locality_domain
