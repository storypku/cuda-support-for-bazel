load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//tools:storydev.bzl", "storydev_gpu_library")

cc_library(
    name = "cuda_exception",
    hdrs = ["cuda_exception.h"],
)

cc_library(
    name = "cuda_helper_timer",
    hdrs = ["cuda_helper_timer.h"],
    deps = [":cuda_exception"],
)

cuda_library(
    name = "reduction_cg_kernel",
    srcs = ["reduction_cg_kernel.cu"],
    hdrs = ["reduction_cg_kernel.h"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "reduction_cg",
    srcs = ["reduction_cg.cpp"],
    deps = [
        ":cuda_helper_timer",
        ":reduction_cg_kernel",
    ],
)
