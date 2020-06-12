load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "argsParser",
    hdrs = ["argsParser.h"],
)

cc_library(
    name = "half",
    hdrs = ["half.h"],
)

cuda_library(
    name = "buffers",
    hdrs = ["buffers.h"],
    deps = [
        ":half",
    ],
)

cuda_library(
    name = "common",
    hdrs = ["common.h"],
    deps = [
        ":logger",
    ],
)

cuda_library(
    name = "logger",
    srcs = ["logger.cpp"],
    hdrs = ["logger.h"],
    deps = [
        ":logging",
    ],
)

cuda_library(
    name = "logging",
    hdrs = ["logging.h"],
    deps = [
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)

cc_binary(
    name = "sampleMNIST",
    srcs = ["sampleMNIST.cpp"],
    deps = [
        ":argsParser",
        ":buffers",
        ":common",
        ":logger",
        ":logging",
        "@local_config_cuda//cuda:cudart",
        "@local_config_tensorrt//:tensorrt",
        "@local_config_tensorrt//:tensorrt_headers",
    ],
)
