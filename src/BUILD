load("@rules_cc//cc:defs.bzl", "cc_binary")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("//tools:storydev.bzl", "storydev_gpu_kernel_library", "storydev_gpu_library", "storydev_kernel_library")

cuda_library(
    name = "hello_world_cuda_lib",
    srcs = ["hello_world.cu"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world",
    deps = [
        ":hello_world_cuda_lib",
    ],
)

storydev_gpu_library(
    name = "hello_world_gpu_lib",
    srcs = ["hello_world.cu"],
    copts = ["-DWHO=1"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world_gpu",
    copts = ["-DWHO=1"],
    deps = [
        ":hello_world_gpu_lib",
    ],
)

storydev_kernel_library(
    name = "hello_world_kernel_lib",
    srcs = ["hello_world.cu"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world_kernel",
    copts = ["-DWHO=2"],
    deps = [
        ":hello_world_kernel_lib",
    ],
)

storydev_gpu_kernel_library(
    name = "hello_world_gpu_kernel_lib",
    srcs = ["hello_world.cu"],
    copts = ["-DWHO=3"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world_gpu_kernel",
    deps = [
        ":hello_world_gpu_kernel_lib",
    ],
)
