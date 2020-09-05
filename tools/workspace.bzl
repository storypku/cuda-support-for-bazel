# External dependencies that can be loaded in WORKSPACE files.

load("//third_party/gpus:cuda_configure.bzl", "cuda_configure")

# load("//third_party/py:python_configure.bzl", "python_configure")
load("//third_party/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")

# Define all external repositories required by storydev
def storydev_repositories():
    cuda_configure(name = "local_config_cuda")
    tensorrt_configure(name = "local_config_tensorrt")
    # python_configure(name = "local_config_python")
