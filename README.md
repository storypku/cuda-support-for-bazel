# cuda-support-in-bazel (WIP)

## Introduction

As of Bazel 3.2.0, there is still no native CUDA support in Bazel (Refer to [Bazel issue: Native CUDA Support](https://github.com/bazelbuild/bazel/issues/6578) to see why). As such, I extracted CUDA support for Bazel code from the [Tensorflow](https://github.com/tensorflow/tensorflow.git) project.

The mechanism behind CUDA support for Bazel was Bazel's custom toolchain support. Refer to [tutorial-cc-toolchain-config](https://docs.bazel.build/versions/master/tutorial/cc-toolchain-config.html) for an thorough understanding.

## Warnings Ahead

Note that CUDA support for Bazel IN THIS REPO was tailored to run on Linux ONLY. Windows and MacOS support are stripped off which are present in the original TensorFlow repo.

As to CPU architectures, ONLY `x86_64` and `aarch64` support are reserved for the editor's own need. If you need support for a different cpu arch, please refer to [Tensorflow](https://github.com/tensorflow/tensorflow.git).

The editor also adjusted default CUDA compute capability settings. Only newer NVidia GPU Cards with SM values = ["6.0", "6.1", "7.0", "7.2", "7.5", ...] are supported, and requires CUDA Toolkit with version >= 9.0.

## Status

**It works!**

As of Fri May 29 08:12:47 CST 2020, this project is still a Work-In-Progress, with a few bugs to be fixed.
- TensorRT support was not tailored yet
- NCCL support was not tailored yet.

### Stripped Modules/Dependencies
- [gemmlowp](https://github.com/google/gemmlowp)
- [mkl](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)

### References
- TF: third_party/mlir/tblgen.bzl
- TF: tensorflow/workspace.bzl
- TF: tensorflow/tensorflow.bzl
- Using Starlark Debugger: https://github.com/bazelbuild/vscode-bazel#using-the-starlark-debugger
- [Deprecated] https://github.com/Hibbert-pku/bazel_nvcc.git

## How to Run

### Step 1: Edit `.storydev.bazelrc` to adjust your Python and CUDA Toolkit versions manually.
TODO(storypku): Provides a `bootstrap` script to configure this, just like Tensorflow's `configure`.

### Step 2: Determine your nVidia GPU's compute capability (abbr. SM), or a list of SMs you want to support. And perform a **find-and-replace** operation in this REPO.

TODO(storypku): Check and set SM values (E.g., `sm_61`, `compute_61`) when running `bootstrap`.

### Step 3: Install Bazel 3.0+.

Ref: [GitHub Bazel Release Page](https://github.com/bazelbuild/bazel/releases)

The following was tested on an Ubuntu 18.04 x86_64 machine with NVidia GTX 1070, using Bazel 3.2.0.

### Step 4: Run the `hello-world` example to check if everything works fine.
```
bazel run //src:hello_world
```

TODO(storypku): More examples, esp. shared library use cases.

## Welcome On Board with Bazel!

