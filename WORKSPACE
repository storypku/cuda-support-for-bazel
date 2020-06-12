workspace(name = "storydev")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz"],
)

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("//tools/gpus:cuda_configure.bzl", "cuda_configure")
load("//tools/tensorrt:tensorrt_configure.bzl", "tensorrt_configure")

cuda_configure(name = "local_config_cuda")

tensorrt_configure(name = "local_config_tensorrt")

http_archive(
    name = "rules_cc",
    sha256 = "cf3b76a90c86c0554c5b10f4b160f05af71d252026b71362c4674e2fb9936cf9",
    strip_prefix = "rules_cc-01d4a48911d5e7591ecb1c06d3b8af47fe872371",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/01d4a48911d5e7591ecb1c06d3b8af47fe872371.zip",
    ],
)
