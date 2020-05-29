load("@rules_cc//cc:defs.bzl", "cc_binary")
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

cuda_library(
    name = "vector_add_gpu_lib",
    srcs = ["vector_add.cu"],
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "vector_add",
    deps = [
        ":vector_add_gpu_lib",
    ],
)
