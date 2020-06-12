load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "fp16_emu_lib",
    srcs = ["fp16_emu.cpp"],
    hdrs = ["fp16_emu.h"],
)

cuda_library(
    name = "error_util",
    hdrs = ["error_util.h"],
)

cuda_library(
    name = "fp16_dev_lib",
    srcs = ["fp16_dev.cu"],
    hdrs = ["fp16_dev.h"],
    deps = [
        ":error_util",
        ":fp16_emu_lib",
    ],
)

cuda_library(
    name = "conv_sample_lib",
    srcs = ["conv_sample.cpp"],
    copts = ["-Wunused-but-set-variable"],
    deps = [
        ":fp16_dev_lib",
        ":fp16_emu_lib",
    ],
)

cc_binary(
    name = "conv_sample",
    srcs = ["conv_sample.cpp"],
    copts = ["-Wunused-but-set-variable"],
    deps = [
        ":conv_sample_lib",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
        "@local_config_cuda//cuda:cudnn",
    ],
)
