load("@rules_cc//cc:defs.bzl", "cc_binary")

#@unused
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

# WARNING: `data` attribute

cuda_library(
    name = "multiHeadAttention_lib",
    srcs = ["multiHeadAttention.cpp"],
    hdrs = ["multiHeadAttention.h"],
    linkopts = ["-lm"],
    deps = [
        "@local_config_cuda//cuda",
        "@local_config_cuda//cuda:cudnn",
    ],
)

cc_binary(
    name = "multiHeadAttention",
    deps = [
        ":multiHeadAttention_lib",
    ],
)

cc_binary(
    name = "multiHeadAttention2",
    srcs = [
        "multiHeadAttention.cpp",
        "multiHeadAttention.h",
    ],
    linkopts = ["-lm"],
    deps = [
        "@local_config_cuda//cuda",
        "@local_config_cuda//cuda:cudnn",
    ],
)
