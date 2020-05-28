#buildifier: @unused
load("//tools/platform:cuda_build_defs.bzl", "if_cuda_is_configured")

#buildifier: @unused
load("//tools:apollo.bzl", "apollo_kernel_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "hello_world",
    srcs = ["hello_world.cu"],
    linkopts = [
        "-Wl,-rpath,../local_config_cuda/cuda/lib64",
        "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
    ],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

#cc_library(
#    name = "hello_world_lib",
#    srcs = ["hello_world.cu.cc"],
#    deps = [
#        ":hello_world_gpu_lib",
#        "@local_config_cuda//cuda",
#    ],
#    alwayslink = True,
#)
