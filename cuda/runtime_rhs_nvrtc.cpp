// Optional NVRTC helper for runtime-specialized MOSAIC-ODE RHS kernels.
// This file is compiled only when WEAVEODE_ENABLE_NVRTC=ON.

#if defined(WEAVEODE_ENABLE_NVRTC)
#include <nvrtc.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace weaveode::cuda_jit {

struct PtxResult {
    std::string PTX;
    std::string log;
};

PtxResult compile_to_ptx(const std::string &cuda_source, const std::string &name,
                         const std::vector<std::string> &options) {
    nvrtcProgram program{};
    nvrtcResult status = nvrtcCreateProgram(&program, cuda_source.c_str(), name.c_str(), 0, nullptr, nullptr);
    if (status != NVRTC_SUCCESS) {
        throw std::runtime_error("nvrtcCreateProgram failed");
    }
    std::vector<const char *> raw_options;
    raw_options.reserve(options.size());
    for (const auto &option : options) raw_options.push_back(option.c_str());
    status = nvrtcCompileProgram(program, (int)raw_options.size(), raw_options.data());
    size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    if (log_size) nvrtcGetProgramLog(program, log.data());
    if (status != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&program);
        throw std::runtime_error("nvrtc compilation failed: " + log);
    }
    size_t ptx_size = 0;
    nvrtcGetPTXSize(program, &ptx_size);
    std::string ptx(ptx_size, '\0');
    nvrtcGetPTX(program, ptx.data());
    nvrtcDestroyProgram(&program);
    return {std::move(ptx), std::move(log)};
}

}  // namespace weaveode::cuda_jit
#endif
