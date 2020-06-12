load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

# WARNING: `data` attribute

cuda_library(
    name = "fp16_emu_lib",
    srcs = ["fp16_emu.cpp"],
    hdrs = ["fp16_emu.h"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cuda_library(
    name = "RNN_example_lib",
    srcs = ["RNN_example.cu"],
    deps = [
        ":fp16_emu_lib",
        "@local_config_cuda//cuda:cudnn",
        #       "@local_config_cuda//cuda:cudnn_header",
    ],
)

cc_binary(
    name = "RNN_example",
    deps = [
        ":RNN_example_lib",
    ],
)
