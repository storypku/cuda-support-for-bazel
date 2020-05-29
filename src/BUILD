load("@rules_cc//cc:defs.bzl", "cc_binary")
load("//tools:storydev.bzl", "storydev_gpu_library")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cuda_library(
    name = "hello_world_gpu_lib",
    srcs = ["hello_world.cu"],
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

storydev_gpu_library(
    name = "hello_world_agpu_lib",
    srcs = ["hello_world.cu"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "hello_world2",
    deps = [
        ":hello_world_agpu_lib",
    ],
)
