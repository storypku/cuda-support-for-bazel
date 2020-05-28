# cuda-support-in-bazel (WIP)

## Introduction

As of Bazel 3.1.0, there is still no native CUDA support in Bazel. See
[Bazel issue: Native CUDA Support](https://github.com/bazelbuild/bazel/issues/6578).

As such, I extracted CUDA support for Bazel code from the [Tensorflow](https://github.com/tensorflow/tensorflow.git) Project.

The mechanism behind `CUDA support for Bazel` was Bazel's custom toolchain support. Refer to [tutorial-cc-toolchain-config](https://docs.bazel.build/versions/master/tutorial/cc-toolchain-config.html) for an thorough understanding.

## Warnings Ahead

Note that `CUDA support for Bazel` IN THIS REPO was tailored to run on Linux ONLY.
The "editor" stripped off Windows and MacOS support which are present in the original TensorFlow repo.

Note also the "editor" adjusted compute capabilies support. Only SM = ["6.0", "6.1", "7.0", "7.2", "7.5"] cards are supported, and requires CUDA version >= 9.0.

## Status

As of Fri May 22 22:22:59 CST 2020, this project is still a Work-In-Progress, with a few bugs to be fixed.

- https://github.com/tensorflow/tensorflow/issues/39759
- `tools/find_cuda_config.py` compress/decompress, see code.
- TensorRT support was not tailored yet
- NCCL support was not tailored yet.

### Stripped Modules/Dependencies
- [gemmlowp](https://github.com/google/gemmlowp)
- [mkl](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html)
- nccl
- tensorflow/workspace.bzl

## Welcome on board with Bazel~


