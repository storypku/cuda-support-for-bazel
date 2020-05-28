load("@rules_cc//cc:defs.bzl", "cc_binary")

#buildifier: @unused
load("//tools:apollo.bzl", "apollo_gpu_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "hello_world_gpu_lib",
    srcs = ["hello_world.cu"],
    #linkopts = [
    #    "-Wl,-rpath,../local_config_cuda/cuda/lib64",
    #    "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
    #],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world",
    deps = [
        ":hello_world_gpu_lib",
    ],
)

#apollo_gpu_library(
#    name = "hello_world_agpu_lib",
#    srcs = ["hello_world.cu"],
#    deps = [
#        "@local_config_cuda//cuda:cudart",
#    ],
#)
#
#cc_binary(
#    name = "hello_world2",
#    deps = [
#        ":hello_world_agpu_lib",
#    ],
#)
