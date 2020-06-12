load("@rules_cc//cc:defs.bzl", "cc_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "argsParser",
    hdrs = ["argsParser.h"],
)

cuda_library(
    name = "buffers",
    hdrs = ["buffers.h"],
)

cuda_library(
    name = "logger",
    hdrs = ["logger.h"],
    deps = [
        ":logging",
    ],
)

cuda_library(
    name = "logging",
    hdrs = ["logging.h"],
    #    deps = [
    #        "libnvinfer-dev",
    #    ],
)
