#buildifier: @unused
load("//tools/platform:cuda_build_defs.bzl", "if_cuda_is_configured")
load("//tools/platform:rules_cc.bzl", "cc_binary")
load("//tools:apollo.bzl", "apollo_gpu_library")

cc_binary(
    name = "hello_world",
    deps = [
        ":hello_world_gpu_lib",
    ],
)

apollo_gpu_library(
    name = "hello_world_gpu_lib",
    srcs = ["hello_world.cu"],
    deps = [
        "@local_config_cuda//cuda",
    ],
)
