# This file is adapted from tensorflow.git/tensorflow/tensorflow.bzl

load("//tools/platform:rules_cc.bzl", "cc_binary", "cc_library", "cc_test")
#load("@local_config_tensorrt//:build_defs.bzl", "if_tensorrt")

load("//tools/platform:cuda_build_defs.bzl", "if_cuda_is_configured")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "cuda_library",
    "if_cuda",
)

# placeholder for mkl, Intel Math Kernel Library

# Sanitize a dependency so that it works correctly from code that includes
# Apollo as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def if_nvcc(a):
    return select({
        "@local_config_cuda//cuda:using_nvcc": a,
        "//conditions:default": [],
    })

def if_cuda_is_configured_compat(x):
    return if_cuda_is_configured(x)

# Given a source file, generate a test name.
# i.e. "common_runtime/direct_session_test.cc" becomes
#      "common_runtime_direct_session_test"
def src_to_test_name(src):
    return src.replace("/", "_").replace(":", "_").split(".")[0]

def full_path(relative_paths):
    return [native.package_name() + "/" + relative for relative in relative_paths]

def if_x86_mode(a):
    return select({
        clean_dep("//tools/platform:x86_mode"): a,
        "//conditions:default": [],
    })

# Linux systems may required -lrt linker flag for e.g. clock_gettime
# see https://github.com/tensorflow/tensorflow/issues/15129
def lrt_if_needed():
    lrt = ["-lrt"]
    return select({
        clean_dep("//tools/platform:aarch64_mode"): lrt,
        clean_dep("//tools/platform:x86_mode"): lrt,
        "//conditions:default": [],
    })

#def apollo_opts_nortti():
#return [
#    "-fno-rtti",
#    "-DGOOGLE_PROTOBUF_NO_RTTI",
#    "-DGOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
#]
#def apollo_defines_nortti():
#    return [
#        "GOOGLE_PROTOBUF_NO_RTTI",
#        "GOOGLE_PROTOBUF_NO_STATIC_INITIALIZER",
#    ]

# if_tensorrt(["-DGOOGLE_TENSORRT=1"]) +
#def apollo_copts(allow_exceptions = False):
#    return [
#        "-DEIGEN_AVOID_STL_ARRAY",
#        "-Wno-sign-compare",
#        "-ftemplate-depth=900",
#    ] + if_cuda(["-DGOOGLE_CUDA=1"]) +
#    if_nvcc(["-DTENSORFLOW_USE_NVCC=1"]) +
#    if_x86_mode(["-msse3"]) +
#    (["-fno-exceptions"] if not allow_exceptions else []) +
#    ["-pthread"]

def apollo_gpu_library(deps = None, cuda_deps = None, copts = [], **kwargs):  #apollo_copts()
    """Generate a cc_library with a conditional set of CUDA dependencies.

      When the library is built with --config=cuda:

      - Both deps and cuda_deps are used as dependencies.
      - The cuda runtime is added as a dependency (if necessary).
      - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts.
      - In addition, when the library is also built with TensorRT enabled, it
          additionally passes -DGOOGLE_TENSORRT=1 to the list of copts.

      Args:
      - cuda_deps: BUILD dependencies which will be linked if and only if:
          '--config=cuda' is passed to the bazel command line.
      - deps: dependencies which will always be linked.
      - copts: copts always passed to the cc_library.
      - kwargs: Any other argument to cc_library.
      """
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    cc_library(
        deps = deps + if_cuda_is_configured_compat(cuda_deps + [
            #   clean_dep("//tensorflow/stream_executor/cuda:cudart_stub"),
            "@local_config_cuda//cuda:cuda_headers",
        ]),
        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"])),
        #if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
        **kwargs
    )

# terminology interoperability: saving apollo_cuda_* definition for convenience
def apollo_cuda_library(*args, **kwargs):
    apollo_gpu_library(*args, **kwargs)

def apollo_gpu_binary(deps = None, cuda_deps = None, copts = [], **kwargs):  #apollo_copts()
    """Generate a cc_binary with a conditional set of CUDA dependencies.
      When the binary is built with --config=cuda:
      Args:
      - cuda_deps: BUILD dependencies which will be linked if and only if:
          '--config=cuda' is passed to the bazel command line.
      - deps: dependencies which will always be linked.
      - copts: copts always passed to the cc_binary.
      - kwargs: Any other argument to cc_binary
      """
    if not deps:
        deps = []
    if not cuda_deps:
        cuda_deps = []

    kwargs["features"] = kwargs.get("features", []) + ["-use_header_modules"]
    cc_binary(
        deps = deps + if_cuda_is_configured_compat(cuda_deps + [
            #   clean_dep("//tensorflow/stream_executor/cuda:cudart_stub"),
            "@local_config_cuda//cuda:cuda_headers",
        ]),
        copts = (copts + if_cuda(["-DGOOGLE_CUDA=1"])),
        #if_tensorrt(["-DGOOGLE_TENSORRT=1"])),
        **kwargs
    )

def apollo_cuda_binary(*args, **kwargs):
    apollo_gpu_binary(*args, **kwargs)
