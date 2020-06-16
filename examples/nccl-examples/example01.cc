#include <cstdio>
#include <cstdlib>

#include "cuda_runtime.h"
#include "nccl.h"

#ifndef NUM_GPU
#define NUM_GPU 2
#endif

#define CUDACHECK(cmd)                                              \
  do {                                                              \
    cudaError_t e = cmd;                                            \
    if (e != cudaSuccess) {                                         \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

#define NCCLCHECK(cmd)                                              \
  do {                                                              \
    ncclResult_t r = cmd;                                           \
    if (r != ncclSuccess) {                                         \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
             ncclGetErrorString(r));                                \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  } while (0)

int main(int argc, char* argv[]) {
  ncclComm_t comms[NUM_GPU];

  // managing NUM_GPU devices
  int nDev = NUM_GPU;
  int size = 32 * 1024 * 1024;
  int devs[NUM_GPU] = { 0 };
  for (int i = 0; i < nDev; ++i) {
      devs[i] = i;
  }

  // allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t) * nDev);

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s + i));
  }

  // initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  // calling NCCL communication API. Group API is required when using
  // multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size,
                            ncclFloat, ncclSum, comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  // synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  // free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  // finalizing NCCL
  for (int i = 0; i < nDev; ++i) ncclCommDestroy(comms[i]);

  printf("Success \n");
  return 0;
}
