#buildifier: @unused
load("//tools/platform:cuda_build_defs.bzl", "if_cuda_is_configured")
load("//tools/platform:rules_cc.bzl", "cc_binary")

#buildifier: @unused
load("//tools:apollo.bzl", "apollo_kernel_library")
load("//tools:apollo.bzl", "apollo_cuda_library", "apollo_gpu_library")

#cc_binary(
#    name = "hello_world",
#    deps = [
#        ":hello_world_lib",
#    ],
#)

#cc_library(
#    name = "hello_world_lib",
#    srcs = ["hello_world.cu.cc"],
#    deps = [
#        ":hello_world_gpu_lib",
#        "@local_config_cuda//cuda",
#    ],
#    alwayslink = True,
#)

apollo_gpu_library(
    name = "hello_world_gpu_lib",
    srcs = ["hello_world.cu"],
    deps = [
        "@local_config_cuda//cuda",
    ],
    alwayslink = True,
)
