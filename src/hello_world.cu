#include<stdio.h>
#include<stdlib.h>

__global__ void print_from_gpu(void) {
    printf("Device: Hello World from thread [%d,%d]\n",
            threadIdx.x, blockIdx.x);
#if defined(WHO)
    printf("Device: Brought to you by %d\n", WHO);
#endif
}

int main(void) {
    printf("Host: Hello World!\n");
#if defined(WHO)
    printf("Host: Brought to you by %d\n", WHO);
#endif
    print_from_gpu <<< 1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

