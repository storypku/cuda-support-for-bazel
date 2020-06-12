// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

//
// This example demonstrates how to use CUDNN library calls cudnnConvolutionForward,
// cudnnConvolutionBackwardData, and cudnnConvolutionBackwardFilter with the option
// to enable Tensor Cores on Volta with cudnnSetConvolutionMathType.
//
// 1. Make sure cuda and cudnn are installed in the same directory.
//
// 2. Run make from the directory of the sample specifying the cuda installation path:
//        make CUDA_PATH=<cuda installation path>
//
// 3. Use the following arguments to run sample with different convolution parameters:
//        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c512 -h28 -w28 -k128 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c512 -h28 -w28 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c512 -h28 -w28 -k256 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c256 -h14 -w14 -k256 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -c256 -h14 -w14 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c1024 -h14 -w14 -k256 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c1024 -h14 -w14 -k512 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -c512 -h7 -w7 -k512 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -c512 -h7 -w7 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//
// 4. Use the following arguments to run sample with int8x4 and int8x32 benchmarks:
//        -filterFormat2 -n1 -c512 -h100 -w100 -k512 -r8 -s8 -pad_h0 -pad_w0 -u1 -v1 -b
//        -filterFormat2 -n1 -c4096 -h64 -w64 -k512 -r4 -s4 -pad_h1 -pad_w1 -u1 -v1 -b
//        -filterFormat2 -n1 -c512 -h100 -w100 -k512 -r8 -s8 -pad_h1 -pad_w1 -u1 -v1 -b
//        -filterFormat2 -n1 -c512 -h128 -w128 -k512 -r13 -s13 -pad_h1 -pad_w1 -u1 -v1 -b
//
// 5. Use the following additional arguments to run the layer with a different setup:
//        -mathType1     : enable Tensor Cores on Volta.
//        -dataType0     : Data is represented as FLOAT
//        -dataType1     : Data is represented as HALF
//        -dataType2     : Data is represented as INT8x4
//        -dataType3     : Data is represented as INT8x32
//        -dgrad         : run cudnnConvolutionBackwardData() instead of cudnnConvolutionForward().
//        -wgrad         : run cudnnConvolutionBackwardFilter() instead of cudnnConvolutionForward().
//        -n<int>        : mini batch size. (use -b with large n)
//        -b             : benchmark mode. Bypass the CPU correctness check.
//        -filterFormat0 : Use tensor format CUDNN_TENSOR_NCHW (Default).
//        -filterFormat1 : Use tensor format CUDNN_TENSOR_NHWC.
//        -filterFormat2 : Use tensor format CUDNN_TENSOR_NCHW_VECT_C. Using this
//                         format switches to int8x4 and int8x32 testing
//        -convBiasAct   : Implement bias and activation operation for convolution .
//
// 6. Note that changing the "-filterFormat" flag will automatically switch to valid data types for
//    that format. CUDNN_TENSOR_NCHW and CUDNN_TENSOR_NHWC support single and half precision
//    tests, while CUDNN_TENSOR_NCHW_VECT_C supports int8x4 and int8x32 tests.
//
//7. "-fold" flag is useful for strided cases, FFT algorithm is chosen for demo purposes, but it can be applied to other algorithms as well
//
// 8. "-reorderMode" flag is used to reorder the filter and bias matrices for convolution. Use the following arguments to run the 
//  convolution with reordered filter and bias matrices.
//
//			-filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c64 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c64 -h32 -w32 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//          -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k128 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//
//
// 9. "-convBiasAct" flag is used to implement "bias"mechanism for convolution.
//
//        -convBiasAct -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -convBiasAct -c512 -h28 -w28 -k128 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -convBiasAct -c512 -h28 -w28 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -convBiasAct -c512 -h28 -w28 -k256 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -convBiasAct -c256 -h14 -w14 -k256 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -convBiasAct -c256 -h14 -w14 -k1024 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -convBiasAct -c1024 -h14 -w14 -k256 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -convBiasAct -c1024 -h14 -w14 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -convBiasAct -c1024 -h14 -w14 -k512 -r1 -s1 -pad_h0 -pad_w0 -u2 -v2
//        -convBiasAct -c512 -h7 -w7 -k512 -r3 -s3 -pad_h1 -pad_w1 -u1 -v1
//        -convBiasAct -c512 -h7 -w7 -k2048 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//        -convBiasAct -c2048 -h7 -w7 -k512 -r1 -s1 -pad_h0 -pad_w0 -u1 -v1
//		  -convBiasAct -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c64 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k32 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c32 -h16 -w16 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c64 -h32 -w32 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k64 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//        -convBiasAct -filterFormat2 -dataType3 -n5 -c128 -h16 -w16 -k128 -r5 -s5 -reorderMode1 -pad_h0 -pad_w0 -u1 -v1 -b
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <float.h>
#include <algorithm>

#include <cudnn.h>
#include "fp16_dev.h"
#include "fp16_emu.h"

#define SWITCH_CHAR             '-'
#define THRESHOLD               2.0e-2

#if defined(__linux__)
#include <stddef.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
static double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#elif defined(__QNX__)
#include <time.h>
static double second (void)
{   
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return ((double)tp.tv_sec + (double)tp.tv_nsec / 1000000000.0);
}
#else
#error unsupported platform
#endif

//Generate uniform numbers [0,1)
static void initImage(float* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
        image[index] = float(seed)*2.3283064e-10; //2^-32
    }
}

static void initImage(half1* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
        image[index] = cpu_float2half_rn(float(seed)*2.3283064e-10); //2^-32
    }
}
//Currently set to generate uniform integers [-2, 2] to avoid int8 overflow
static void initImage(int8_t* image, int imageSize) {
    static unsigned seed = 123456789;
    for (int index = 0; index < imageSize; index++) {
        seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
        //Takes floats from [0, 1), scales and casts to ints from [0, 4], then subtracts from 2
        image[index] = 2 - (int8_t)(5 * float(seed)*2.3283064e-10); //2^-32
    }
}

static void initImagePadded(int8_t* image, int dimA[], int dimPadded[], int stridePadded[], cudnnDataType_t dataType) {
    static unsigned seed = 123456789;
    int resizeFactor = (dataType == CUDNN_DATA_INT8x4) ? 4 : 32;
    int totalSize = dimPadded[0] * dimPadded[1] * dimPadded[2] * dimPadded[3];

    // #pragma omp parallel for
    for (int i = 0; i < totalSize; i++) {
        int n = (i / stridePadded[0]) % dimPadded[0];
        int c1 = (i / (stridePadded[1] * resizeFactor)) % (dimPadded[1] / resizeFactor);
        int c2 = i % resizeFactor;
        int c = c1 * resizeFactor + c2;
        if (n < dimA[0] && c < dimA[1]) {
			seed = ( 1103515245 * seed + 12345 ) & 0xffffffff;
            image[i] = 2 - (int8_t)(5 * float(seed)*2.3283064e-10); //2^-32
        } else {
            image[i] = 0;
        }
    }
}


static int checkCudaError(cudaError_t code, const char* expr, const char* file, int line) {
    if (code) {
        printf("CUDA error at %s:%d, code=%d (%s) in '%s'", file, line, (int) code, cudaGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

static int checkCudnnError(cudnnStatus_t code, const char* expr, const char* file, int line) {
    if (code)  {
        printf("CUDNN error at %s:%d, code=%d (%s) in '%s'\n", file, line, (int) code, cudnnGetErrorString(code), expr);
        return 1;
    }
    return 0;
}

#define checkCudaErr(...)       do { int err = checkCudaError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__); if (err) goto clean; } while (0)

#define checkCudnnErr(...)      do { int err = checkCudnnError(__VA_ARGS__, #__VA_ARGS__, __FILE__, __LINE__);  if (err) goto clean; } while (0)

static void printPerf( double cudaTime, double cudaGflops, double cudaBandwithGb,
                      const char *cpuLib, double cpuTime,  double cpuGflops, double cpuBandwithGb) {
    printf( "^^^^ CUDA : elapsed = %g sec,  ",  cudaTime );
    if (cudaGflops > 0)    printf( "Gflops = %.3f ",      cudaGflops );
    if (cudaBandwithGb > 0) printf( "Bandwidth = %.3f ",  cudaBandwithGb );
    printf( "\n");

    if (cpuLib) {
        printf( "^^^^%s : elapsed = %g sec, ",  cpuLib, cpuTime );
        if (cpuGflops > 0)    printf( "Gflops = %.3f ",      cpuGflops );
        if (cpuBandwithGb > 0) printf( "Bandwidth = %.3f, ",  cpuBandwithGb );
        printf( "Speedup %.2f\n",  cpuTime/cudaTime );

    }
}

static void generateStrides(const int* dimA, int* strideA, int nbDims, cudnnTensorFormat_t filterFormat) {
    //For INT8x4 and INT8x32 we still compute standard strides here to input
    //into the cuDNN functions. We will manually scale by resizeFactor in the cpu ref.
    if (filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        strideA[nbDims-1] = 1 ;
        for(int d = nbDims-2 ; d >= 0 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
    } else {
        //Here we assume that the format is CUDNN_TENSOR_NHWC
        strideA[1] = 1;
        strideA[nbDims-1] = strideA[1]*dimA[1];
        for(int d = nbDims-2 ; d >= 2 ; d--) {
            strideA[d] = strideA[d+1] * dimA[d+1] ;
        }
        strideA[0] = strideA[2]*dimA[2];
    }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void lin2dim(int id, int* ids, const int* dims, int length) {
    int idrem = id ;
    int prod  = 1 ; // accumulates the product of the dimensions
    for(int i = length-1; i >= 0; i--) {
        ids[i] = (idrem / prod) % dims[i] ;
        idrem = id - ids[i] * prod ;
        prod *= dims[i] ;
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
static int dim2lin(const int* ids, const int* strides, int length) {
    int res = 0 ;
    for(int i = 0 ; i < length ; i++) {
        res += ids[i] * strides[i];
    }
    return res ;
}

static float doFma(float fval, float ival, float tmp) {
    return fval*ival+tmp;
}

static float doFma(half1 fval, half1 ival, float tmp) {
    return cpu_half2float(fval)*cpu_half2float(ival)+tmp;
}

static int32_t doFma(int8_t fval, int8_t ival, int32_t tmp) {
    return int32_t(fval)*int32_t(ival)+tmp;
}



//Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(float fval, float ival, int32_t tmp) {
    return 0;
}
//Garbage function, resolves overloaded function ambiguity for an invalid type combination
static int32_t doFma(half1 fval, half1 ival, int32_t tmp) {
    return 0;
}
//Garbage function, resolves overloaded function ambiguity for an invalid type combination
static float doFma(int8_t fval, int8_t ival, float tmp) {
    return 0;
}

static void doEpilog(float *out, int idx, float alphaAcc, float beta) {
    if( beta == 0.f ) {
        out[idx] = alphaAcc;
    } else {
        out[idx] = alphaAcc + out[idx]*beta;
    }
}

static void doEpilog(half1 *out, int idx, float alphaAcc, float beta) {
    if( beta == 0.f ) {
        out[idx] = cpu_float2half_rn(alphaAcc);
    } else {
        out[idx] = cpu_float2half_rn(alphaAcc + cpu_half2float(out[idx])*beta);
    }
}

static void doEpilog(int8_t *out, int idx, int32_t alphaAcc, float beta) {
    int32_t val;
    if( beta == 0.f ) {
        val = alphaAcc;
    } else {
        val = alphaAcc + out[idx]*beta;
    }
    //Properly handle overflow errors in the same way cuDNN does
    if (val > 127) {
        val = 127;
    } else if (val < -128) {
        val = -128;
    }
    out[idx] = val;
}

// Y = ReLU(alpha1*conv(X) + alpha2*Z + bias)
template <typename T_ELEM, typename T_MATH>
static void
addTensorBiasAct_ref(int DIM,
                     const int*   outDims,
                     const int*   outStride,
                     const int*   biasStride,
                     T_ELEM *dest,
                     T_ELEM *Z,
                     const float *bias,
                     double ceil,
                     double floor,
                     float alpha,
                     float beta,
                     cudnnDataType_t dataType) {						 
    bool betaNonZero = (beta != (float)(0.0f));

    for (int image = 0; image < outDims[0]; image++) {
        for (int colmap = 0; colmap < outDims[1]; colmap++) {
             const float data_bias = (bias[colmap * biasStride[1]]);
            T_ELEM *data_inout     = &dest[image * outStride[0] + colmap * outStride[1]];
            T_ELEM *Z_in           = &Z[image * outStride[0] + colmap * outStride[1]];

            for (int i = 0; i < outDims[2]; i++) {
                for (int j = 0; j < ((DIM > 1) ? outDims[3] : 1); j++) {
                    for (int k = 0; k < (DIM == 3 ? outDims[4] : 1); k++) {
                        int idxDest = i * outStride[2] + j * outStride[3];
                        idxDest += (DIM == 3 ? k * outStride[4] : 0);
						
                        float data = convertToFloat((void *)&data_inout[idxDest],dataType);

						data        = (data * alpha);						
						if (betaNonZero) {
							float Z_value = convertToFloat((void *)&Z_in[idxDest],dataType);
                            data = doFma(Z_value, beta, data);
                        }
						
                        data = (data + data_bias);
                        data = std::max(data, (float)(floor));  // clipping floor (implements ReLU)
                        data = std::min(data, (float)(ceil));   // clipping ceil (implements clipped ReLU)

						convertFromFloat(&data, dataType, &data_inout[idxDest]);
                    }
                }
            }
        }
    }
}

//T_ELEM is the type the data is stored in, T_MATH is the type the calculations are done in.
template <typename T_ELEM, typename T_MATH>
static void conv_cpu_ref (
    const T_ELEM* inputData,
    const T_ELEM* filterData,
    T_ELEM*       outputData,
    float        alpha,
    float        beta,
    int          resizeFactor,
    cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   outDims,
    const int*   inStride,
    const int*   outStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims
) {
    int imDims = nbDims - 2 ;

    int filStride[8] = {0} ;
    generateStrides(filDims, filStride, nbDims, filterFormat);

    bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;
    // Number of pixels in output
    int nPixelsOut = 1 ;
    for(int i = 2 ; i < nbDims ; i++)
        nPixelsOut *= outDims[i] ;
    // Number of pixels in filter
    int nPixelsFil = 1 ;
    for(int i = 2 ; i < nbDims ; i++)
        nPixelsFil *= filDims[i] ;
    // Used to store coordinates
    int filIds[8] = {0} ;
    int outIds[8] = {0} ;
    int inIds [8] = {0} ;
    int tmpIds[8] = {0} ;

    // For each image in the output
    for(int ni = 0 ; ni < outDims[0] ; ni++) {
        // For each outer feature layer of the output image
        for(int ki_outer = 0 ; ki_outer < outDims[1] / resizeFactor; ki_outer++) {
            int outputOffset = ni * outStride[0] / resizeFactor + ki_outer * outStride[1] ;
            // For every pixel in this output image's feature layer
            for(int outId = 0 ; outId < nPixelsOut ; outId++) {
                // Get output pixel ids
                lin2dim(outId, outIds, outDims+2, imDims) ; // Skip n and k dimensions
                // Now we get the coordinates in input space of the "top left" corner of the filter: multiply by stride and remove pad
                for(int d = 0 ; d < imDims ; d++) {
                    inIds[d] = outIds[d] * stride[d] - pad[d] ;
                }
                // For each inner feature layer of the output image
                for (int ki_inner = 0; ki_inner < resizeFactor; ki_inner++) {
                    // We prepare to accumulate
                    T_MATH tmp = 0;
                    // For each outer feature layer of the input image and filter
                    for(int ci = 0 ; ci < inDims[1] / resizeFactor; ci++) {
                        int inputOffset = ni * inStride[0] / resizeFactor + ci * inStride[1] ;
                        int filterOffset = (ki_outer * resizeFactor + ki_inner) * filStride[0] / resizeFactor + ci * filStride[1] ;
                        // Now for every pixel in the filter
                        for(int filId = 0 ; filId < nPixelsFil ; filId ++) {
                            // Get the position of the pixel
                            lin2dim(filId, filIds, filDims+2, imDims) ;
                            // Compute the corresponding output pixel
                            // and check whether we are in the padding area on the fly too (not that for convolution, we flip the image patch (equivalent to flipping the filter patch))
                            bool inside = true ;
                            for(int d = 0 ; d < imDims && inside ; d++) {
                                if (isConv) {
                                    tmpIds[d] = inIds[d] + dilation[d] * (filDims[2+d]-1 - filIds[d]) ;
                                } else {
                                    tmpIds[d] = inIds[d] + dilation[d] * filIds[d] ;
                                }
                                inside &= (tmpIds[d] >= 0 && tmpIds[d] < inDims[2+d]) ; // If we are in the padding area: stop and skip computations
                            }
                            if(inside) {
                                int actualTmpId = inputOffset + dim2lin(tmpIds, (inStride)+2, imDims) ;
                                //int actualFilId = filterOffset + filId ;
                                int actualFilId = filterOffset + dim2lin(filIds, (filStride)+2, imDims) ;

                                //For each inner feature layer of the input image and filter
                                for (int i = 0; i < resizeFactor; i++) {
                                    T_ELEM fval = filterData[actualFilId * resizeFactor + i] ;
                                    T_ELEM ival = inputData[actualTmpId * resizeFactor + i] ;
                                    tmp = doFma(fval, ival, tmp);
                                }
                            }
                        }
                    }

                    // Store final result in proper position in output image
                    int actualOutId = outputOffset + dim2lin(outIds, (outStride)+2, imDims) ;
                    doEpilog(outputData, actualOutId * resizeFactor + ki_inner, alpha*tmp, beta);
                }
            }
        }
    }
}

template<typename T_ELEM>
static void dataGrad_cpu_ref (
    const T_ELEM *weight,
    const T_ELEM *top_diff,
    T_ELEM *output,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   outDims,
    const int*   inStride,
    const int*   outStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims,
    cudnnConvolutionMode_t mode)
{

    // Sanity checks
    // output is n x c x h x w
    // diff   is n x k x p x q
    // filter is k x c x r x s
    assert(inDims[0] == outDims[0]); // n
    assert(inDims[1] == filDims[0]); // k
    assert(outDims[1] == filDims[1]); // cactualOutId

    int filStride[8] = {0} ;
    generateStrides(filDims, filStride, nbDims, filterFormat);

    //true for convolution and false for cross-correlation
    bool isConv = (mode == CUDNN_CONVOLUTION)? true : false;

    // For every output pixel (n x c x h x w)
    for(int ni = 0; ni < outDims[0]; ni++) {
        for(int ci = 0; ci < outDims[1]; ci++) {
            for(int hi = 0; hi < outDims[2]; hi++) {
                for(int wi = 0; wi < outDims[3]; wi++) {
                    int outIdx = ni * outStride[0] +
                                 ci * outStride[1] +
                                 hi * outStride[2] +
                                 wi * outStride[3];
                    float val = 0.0;

                    // For every diff channel (k)
                    for(int ki = 0; ki < inDims[1]; ki++) { // Sum over k channels
                        int offset_filter = ki * filStride[0] + ci * filStride[1];
                        int offset_diff   = ni * inStride[0] + ki * inStride[1];
                        // For every pixel if filter (r x s)
                        for(int ri = 0; ri < filDims[2]; ri++) {
                            int p = hi + pad[0];
                            if (isConv){
                                p -= (filDims[2] - 1 - ri) * dilation[0];
                            } else {
                                p -= ri * dilation[0];
                            }
                            if ( p%stride[0] )
                                continue;
                            p/=stride[0];

                            for(int si = 0; si < filDims[3]; si++) {
                                int q = wi + pad[1];
                                // Fetch the value in filter and diff, product and accumulate
                                // So basically, for the convolution, we replace r by dim-1-r and s by dim-1-s to "flip" the filter
                                // We can then just reason in term of correlation
                                if (isConv){
                                    q -= (filDims[3] - 1 - si) * dilation[1];
                                } else {
                                    q -= si * dilation[1];
                                }
                                //Skip if q or p isn't multiple of strides
                                if ( q%stride[1] )
                                    continue;
                                q/=stride[1];
                                int inBounds = ( (p >= 0) && (p < inDims[2]) && (q >= 0) && (q < inDims[3]) );
                                if (inBounds) {
                                int filterIdx = offset_filter + ri * filStride[2] + si * filStride[3];
                                int diffIdx = offset_diff + p * inStride[2] + q * inStride[3];
                                T_ELEM imTmp = top_diff[diffIdx];
                                T_ELEM filTmp = weight[filterIdx];
                                val = doFma(filTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, outIdx, alpha*val, beta);
                }
            }
        }
    }
}
//TODO:Fix this mess
template<typename T_ELEM>
static void weightGrad_cpu_ref(/*const TensorNdTestDesc_t *tensorInputDesc,*/
                                                         const T_ELEM *image,
                                                         /*const TensorNdTestDesc_t *tensorDiffDesc,*/
                                                         const T_ELEM *diffData,
                                                         /*const ConvNdTestDesc_t *convDesc,*/
                                                         /*const TensorNdTestDesc_t *filterOutputDesc,*/
                                                         float alpha,
                                                         float beta,
                                                         T_ELEM *output,
                                                         cudnnTensorFormat_t filterFormat,
    const int*   inDims,
    const int*   filDims,
    const int*   diffDims,
    const int*   inStride,
    const int*   diffStride,
    const int*   stride,
    const int*   pad,
    const int*   dilation,
    int          nbDims )
{
    // Some sanity checks
    // image   is n x c x h x w
    // diff    is n x k x p x q
    // filter  is k x c x r x s
    assert(inDims[0] == diffDims[0]) ;
    assert(inDims[1] == filDims[1]) ;
    assert(diffDims[1]  == filDims[0]) ;

    // Filter stride
    int filterStride[4] ;
    generateStrides(filDims, filterStride, nbDims, filterFormat);

    bool isConv = true; //(CUDNN_CONVOLUTION == mode) ;

    // For every filter pixel (k x c x r x s)
    for(int ci = 0; ci < inDims[1]; ci++) { // Loop over filter output pixels
        for(int ri = 0; ri < filDims[2]; ri++) { //        ^
            for(int si = 0; si < filDims[3]; si++) { //    ^
                for(int ki = 0; ki < filDims[0]; ki++){ // ^
                    int filIdx = ki * filterStride[0] + ci * filterStride[1] + ri * filterStride[2] + si * filterStride[3] ;
                    float val = 0.f ;
                    // For every image (n)
                    for(int ni = 0 ; ni < inDims[0]; ni++) { // Sum over the batch
                        int offset_image  = ni * inStride[0] + ci * inStride[1] ;
                        int offset_diff   = ni * diffStride[0]  + ki * diffStride[1] ;
                        // For every pixel in diff (p x q)
                        for(int pi = 0; pi < diffDims[2] ; pi++ ) { // Sum over the pixels of diff
                            for(int qi = 0; qi < diffDims[3] ; qi++ ) { //  ^
                                // Fetch the value in image and diff, product and accumulate
                                int y = pi * stride[0] - pad[0] ;
                                int x = qi * stride[1] - pad[1] ;
                                // Convolution = Correlation with a flipped filter
                                // So basically, for the convolution, we replace r by dim-1-r and s by dim-1-s to "flip" the filter
                                // We can then just reason in term of correlation
                                if (isConv){
                                    y += (filDims[2] - 1 - ri) * dilation[0] ;
                                    x += (filDims[3] - 1 - si) * dilation[1] ;
                                } else {
                                    // The effect of dilation on the gradient is to start the "zone of influence" of a given pixel further into the image, so dilation
                                    // only produces a shift in x and y
                                    y += ri * dilation[0] ;
                                    x += si * dilation[1] ;
                                }
                                // Image value
                                int inBounds = ((x >=0)&&(x < inDims[3])&&(y >=0)&&(y < inDims[2]));
                                if (inBounds) {
                                int imIdx    = offset_image  + y * inStride[2] + x * inStride[3] ;
                                // Diff value
                                int diffIdx  = offset_diff   + pi * diffStride[2]  + qi * diffStride[3] ;
                                // Prod and accumulate
                                T_ELEM imTmp = image[imIdx] ;
                                T_ELEM diffTmp = diffData[diffIdx];
                                val = doFma(diffTmp, imTmp, val);
                                }
                            }
                        }
                    }
                    doEpilog(output, filIdx, alpha*val, beta);
                }
            }
        }
    }
}


float getError(float dev, float ref) {
    if (ref > 1.0 || ref < -1.0)
        return (dev - ref)/ref;
    else
        return dev - ref;
}

float getError(half1 dev, half1 ref) {
    if (cpu_half2float(ref) > 1.0 || cpu_half2float(ref) < -1.0)
        return (cpu_half2float(dev) - cpu_half2float(ref))/cpu_half2float(ref);
    else
        return cpu_half2float(dev) - cpu_half2float(ref);
}

int8_t getError(int8_t dev, int8_t ref) {
    return dev - ref;
}

static inline int getFwdConvDilatedFilterDim(int filterDim,
                                            int dilation)
{
    return ( (filterDim - 1) * dilation ) + 1 ;
}

static inline int getFwdConvPaddedImageDim(int tensorDim,
                                           int pad)
{
    return tensorDim + (2 * pad) ;
}

static inline int getFwdConvOutputDim( int tensorDim,
                                       int pad,
                                       int filterDim,
                                       int stride,
                                       int dilation)
{
    int p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation))/stride + 1;
    return(p);
}

template <typename T_ELEM>
int doConv(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t   cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    cudnnDataType_t dataType,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int   benchmark) {

    int outsize = outstrideA[0]*outdimA[0];
    T_ELEM* hostOfromdev = (T_ELEM*)calloc (outsize, sizeof(hostO[0]) );

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    void *workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    double start, stop;

    checkCudnnErr ( cudnnGetConvolutionForwardWorkspaceSize(handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc,
        cudnnOdesc, algo, &workSpaceSize) );

    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }

	 start = second();	 
	 checkCudnnErr ( cudnnConvolutionForward (handle_,
				    (void*)(&alpha),
				    cudnnIdesc, devPtrI,
				    cudnnFdesc, devPtrF,
				    cudnnConvDesc,
				    algo,
				    workSpace, workSpaceSize,
				    (void*)(&beta),
				    cudnnOdesc, devPtrO) );
	 checkCudaErr( cudaDeviceSynchronize() );
	
    // host time end
    stop = second();

    printPerf( (stop - start), 0, 0,
                      0, 0, 0, 0);
    checkCudaErr( cudaMemcpy(hostOfromdev, devPtrO, sizeof(hostO[0]) * outsize, cudaMemcpyDeviceToHost) );
    checkCudaErr( cudaDeviceSynchronize() );

	if(!benchmark){
			//Pass in resize factor for the cpu reference solution, this is the number of packed variables in each element of the tensor
		if (filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
			if (dataType == CUDNN_DATA_INT8x4) {//resizeFactor = 4
				conv_cpu_ref<T_ELEM, int32_t>( hostI, hostF, hostO, alpha, beta, 4, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
			} else if (dataType == CUDNN_DATA_INT8x32) {//resizeFactor = 32
				conv_cpu_ref<T_ELEM, int32_t>( hostI, hostF, hostO, alpha, beta, 32, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
			} else {
				printf("CUDNN_TENSOR_NCHW_VECT_C only supports INT8x4 and INT8x32");
				return 1;
			}
		} else {
			conv_cpu_ref<T_ELEM, float>( hostI, hostF, hostO, alpha, beta, 1, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
		}
    
		
		for (int index = 0; index < outsize; index++) {
			float diff = getError(hostOfromdev[index], hostO[index]);
			if (diff < 0) diff = -diff;
				if(diff > THRESHOLD && diff != 1) {
					numErrors++;
				}
				//printf("cuda result is %d, and reference is %d\n", hostOfromdev[index], hostO[index]);
		}
	}
        
clean:
    if (hostOfromdev) free(hostOfromdev);
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

template <typename T_ELEM>
int
doConvBiasAct(cudnnHandle_t handle_,
              T_ELEM* devPtrI,
              T_ELEM* devPtrF,
              T_ELEM* devPtrO,
              float*  devPtrBias,
              T_ELEM* devPtrZ,
              T_ELEM* hostI,
              T_ELEM* hostF,
              T_ELEM* hostO,
              float*  hostBias,
              T_ELEM* hostZ,
              cudnnTensorDescriptor_t cudnnIdesc,
              cudnnFilterDescriptor_t cudnnFdesc,
              cudnnTensorDescriptor_t cudnnOdesc,
              cudnnTensorDescriptor_t cudnnBiasdesc,
              cudnnActivationDescriptor_t activationDesc,
              cudnnConvolutionDescriptor_t cudnnConvDesc,
              float alpha,
              float beta,
              cudnnTensorFormat_t filterFormat,
              cudnnDataType_t dataType,
              const int* dimA,
              const int* filterdimA,
              const int* outdimA,
              const int* biasdimA,
              const int* strideA,
              const int* outstrideA,
              const int* biasstrideA,
              const int* convstrideA,
              const int* padA,
              const int* dilationA,
              const int benchmark) {
    int outsize     = outstrideA[0] * outdimA[0];
    int numErrors = 0;
    void* workSpace = 0;
    size_t workSpaceSize;
    float one = (1.0f);
	float zero = (0.0f);
    T_ELEM* hostOfromdev = (T_ELEM*)calloc(outsize, sizeof(hostO[0]));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    checkCudnnErr(cudnnGetConvolutionForwardWorkspaceSize(
        handle_, cudnnIdesc, cudnnFdesc, cudnnConvDesc, cudnnOdesc, algo, &workSpaceSize));

    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }

    checkCudnnErr(cudnnConvolutionBiasActivationForward(handle_,
                                                        (void*)(&alpha),
                                                        cudnnIdesc,
                                                        devPtrI,
                                                        cudnnFdesc,
                                                        devPtrF,
                                                        cudnnConvDesc,
                                                        algo,
                                                        workSpace,
                                                        workSpaceSize,
                                                        (void*)(&beta),
                                                        cudnnOdesc,
                                                        devPtrZ,
                                                        cudnnBiasdesc,
                                                        devPtrBias,
                                                        activationDesc,
                                                        cudnnOdesc,
                                                        devPtrO));
                                   
                                    
    
    checkCudaErr(cudaDeviceSynchronize());
    checkCudaErr(cudaMemcpy(hostOfromdev, devPtrO, sizeof(hostO[0]) * outsize, cudaMemcpyDeviceToHost));
    if(!benchmark){
			//Pass in resize factor for the cpu reference solution, this is the number of packed variables in each element of the tensor
		if (filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
			if (dataType == CUDNN_DATA_INT8x4) {//resizeFactor = 4
				conv_cpu_ref<T_ELEM, int32_t>( hostI, hostF, hostO, alpha, beta, 4, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
				addTensorBiasAct_ref<T_ELEM, float>(2, outdimA, outstrideA, biasstrideA, hostO, hostZ,hostBias, DBL_MAX, 0, one, zero, dataType);
			} else if (dataType == CUDNN_DATA_INT8x32) {//resizeFactor = 32
				conv_cpu_ref<T_ELEM, int32_t>( hostI, hostF, hostO, alpha, beta, 32, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
			    addTensorBiasAct_ref<T_ELEM, float>(2, outdimA, outstrideA, biasstrideA, hostO, hostZ, hostBias, DBL_MAX, 0, one, zero, dataType);
			} else {
				printf("CUDNN_TENSOR_NCHW_VECT_C only supports INT8x4 and INT8x32");
				return 1;
			}
		} else {
			conv_cpu_ref<T_ELEM, float>( hostI, hostF, hostO, alpha, beta, 1, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
			addTensorBiasAct_ref<T_ELEM, float>(2, outdimA, outstrideA, biasstrideA, hostO, hostZ, hostBias, DBL_MAX, 0, one, zero, dataType);
		}
    
		for (int index = 0; index < outsize; index++) {
			float diff = getError(hostOfromdev[index], hostO[index]);
			if (diff < 0) diff = -diff;
				if(diff > THRESHOLD && diff != 1 && diff != 2) {
					numErrors++;
				}
				// printf("cuda result is %d, and reference is %d\n", (int)hostOfromdev[index], (int)hostO[index]);
		}
	}

clean:
    if (hostOfromdev) free(hostOfromdev);
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

template <typename T_ELEM>
int doDgrad(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t   cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int    benchmark,
    const bool fold,
    cudnnConvolutionMode_t mode) {

    int insize = strideA[0]*dimA[0];
    T_ELEM* hostIfromdev = (T_ELEM*)calloc (insize, sizeof(hostI[0]) );
    // Uses FFT tiling for demonstration purpose when folding is set as true, folding can be used in combination with other algorithms as well 
    cudnnConvolutionBwdDataAlgo_t algo = fold ? CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING : CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    void *workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    double start, stop;

    if(fold){
        start = second();
        // create folded descriptors
        cudnnFilterDescriptor_t foldedFilterDesc;
        cudnnCreateFilterDescriptor(&foldedFilterDesc);
        cudnnTensorDescriptor_t paddedDiffDesc;
        cudnnCreateTensorDescriptor(&paddedDiffDesc);
        cudnnConvolutionDescriptor_t foldedConvDesc;
        cudnnCreateConvolutionDescriptor (&foldedConvDesc);
        cudnnTensorDescriptor_t foldedGradDesc;
        cudnnCreateTensorDescriptor(&foldedGradDesc);
        

        //create empty transform descriptors
        cudnnTensorTransformDescriptor_t filterFoldTransDesc;
        cudnnCreateTensorTransformDescriptor(&filterFoldTransDesc);
        cudnnTensorTransformDescriptor_t diffPadTransDesc;
        cudnnCreateTensorTransformDescriptor(&diffPadTransDesc);
        cudnnTensorTransformDescriptor_t gradFoldTransdesc;
        cudnnCreateTensorTransformDescriptor(&gradFoldTransdesc);
        cudnnTensorTransformDescriptor_t gradUnfoldTransDesc;
        cudnnCreateTensorTransformDescriptor(&gradUnfoldTransDesc);

        //populate folded descriptors
        cudnnGetFoldedConvBackwardDataDescriptors(handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, cudnnIdesc, filterFormat, foldedFilterDesc, paddedDiffDesc, foldedConvDesc, foldedGradDesc, filterFoldTransDesc, diffPadTransDesc, gradFoldTransdesc, gradUnfoldTransDesc);

        float foldAlpha = 1.0f;
        float foldBeta  = 0.0f;

        // Filter
        size_t foldedFilterSize;
        cudnnGetFilterSizeInBytes(foldedFilterDesc, &foldedFilterSize);
        void *foldedFilterData = NULL;
        cudaMalloc(&foldedFilterData, foldedFilterSize);
        if (foldedFilterData == NULL) {
            printf("\n Not enough workspace to allocate foldedFilterData!\n");
            return CUDNN_STATUS_BAD_PARAM;
        }
        cudnnStatus_t status;
        status = cudnnTransformFilter(handle_,
                                        filterFoldTransDesc,
                                        (void *)&foldAlpha,
                                        cudnnFdesc,
                                        (void *)devPtrF,
                                        (void *)&foldBeta,
                                        foldedFilterDesc,
                                        foldedFilterData);
        if (status != CUDNN_STATUS_SUCCESS) {
            return status;
        }
        // Diff
        void *paddedDiffData = NULL;
        size_t paddedDiffDescSize, unpaddedDiffDescSize;
        cudnnGetTensorSizeInBytes(paddedDiffDesc, &paddedDiffDescSize);
        cudnnGetTensorSizeInBytes(cudnnOdesc, &unpaddedDiffDescSize);
        if(paddedDiffDescSize != unpaddedDiffDescSize){    
            cudaMalloc(&paddedDiffData, paddedDiffDescSize);
            if (paddedDiffData == NULL) {
                printf("Not enough workspace to allocate paddedDiffData!\n");
                return CUDNN_STATUS_BAD_PARAM;
            }

            status = cudnnTransformTensorEx(handle_,
                    diffPadTransDesc,
                    (void *)&foldAlpha,
                    cudnnOdesc,
                    (void *)devPtrO,
                    (void *)&foldBeta,
                    paddedDiffDesc,
                    paddedDiffData);
            if (status != CUDNN_STATUS_SUCCESS) {
                return status;
            }
        } else {
            paddedDiffData = (void *)(devPtrO);
        }

        // Folded Grad
        size_t foldedGradDescSize;
        cudnnGetTensorSizeInBytes(foldedGradDesc, &foldedGradDescSize);
        void *foldedGradData = NULL;
        //check for return status of cudaMalloc
        cudaMalloc(&foldedGradData, foldedGradDescSize);
        if (foldedGradData == NULL) {
            printf("Not enough workspace to allocate foldedGradData!\n");
            return CUDNN_STATUS_BAD_PARAM;
        }

        cudaError_t err =
            cudaMemset(foldedGradData, 0, foldedGradDescSize);
        if (err != cudaSuccess) {
            return CUDNN_STATUS_EXECUTION_FAILED;
        }

        //Run dgrad with folded tensors
        checkCudnnErr ( cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, foldedFilterDesc, paddedDiffDesc, foldedConvDesc,
                    foldedGradDesc,algo, &workSpaceSize) );
        printf("\n WORKSPACE = %lu \n", workSpaceSize);
        if (workSpaceSize > 0) {
            cudaMalloc(&workSpace, workSpaceSize);
    }
    checkCudnnErr (cudnnConvolutionBackwardData(handle_,
                                                (void*)(&foldAlpha),
                                                foldedFilterDesc, foldedFilterData,
                                                paddedDiffDesc, paddedDiffData, 
                                                foldedConvDesc,
                                                algo,
                                                workSpace, workSpaceSize,
                                                (void*)(&foldBeta),
                                                foldedGradDesc, foldedGradData));

    // Unfold the grad data        
    status = cudnnTransformTensorEx(handle_,
                                    gradUnfoldTransDesc,
                                    (void *)&alpha,
                                    foldedGradDesc,
                                    foldedGradData,
                                    (void *)&beta,
                                    cudnnIdesc,
                                    devPtrI);
            if (status != CUDNN_STATUS_SUCCESS) {
                return status;
            }
    checkCudaErr(cudaDeviceSynchronize());
    stop = second();
    cudnnDestroyFilterDescriptor(foldedFilterDesc);
    cudnnDestroyTensorDescriptor(paddedDiffDesc);
    cudnnDestroyTensorDescriptor(foldedGradDesc);
    cudnnDestroyConvolutionDescriptor(foldedConvDesc);
    cudnnDestroyTensorTransformDescriptor(filterFoldTransDesc);
    cudnnDestroyTensorTransformDescriptor(diffPadTransDesc);
    cudnnDestroyTensorTransformDescriptor(gradFoldTransdesc);
    cudnnDestroyTensorTransformDescriptor(gradUnfoldTransDesc);


    if(foldedFilterData) cudaFree(foldedFilterData);
    if(paddedDiffData) cudaFree(paddedDiffData);
    if(foldedGradData) cudaFree(foldedGradData);

    }else{
        start = second();
        checkCudnnErr (cudnnGetConvolutionBackwardDataWorkspaceSize(handle_, cudnnFdesc, cudnnOdesc, cudnnConvDesc, cudnnIdesc, algo, &workSpaceSize));
    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    checkCudnnErr (cudnnConvolutionBackwardData(handle_,
                                                (void*)(&alpha),
                                                cudnnFdesc, devPtrF,
                                                cudnnOdesc, devPtrO,
                                                cudnnConvDesc,
                                                algo,
                                                workSpace, workSpaceSize,
                                                (void*)(&beta),
                                                cudnnIdesc, devPtrI));

     checkCudaErr(cudaDeviceSynchronize());
     stop = second();
    }
    //Run reference code and check for errors
    printPerf( stop - start, 0, 0,
                      0, 0, 0, 0);
    checkCudaErr(cudaMemcpy(hostIfromdev, devPtrI, sizeof(hostI[0]) * insize, cudaMemcpyDeviceToHost));
    checkCudaErr(cudaDeviceSynchronize());

    if (!benchmark) {
        dataGrad_cpu_ref<T_ELEM>(hostF, hostO, hostI,  alpha, beta, filterFormat, outdimA, filterdimA, dimA, outstrideA, strideA, convstrideA, padA, dilationA, 4, mode);
        for (int index = 0; index < insize; index++) { // assuming in data is packed
            float diff = getError(hostIfromdev[index], hostI[index]);
            if (diff < 0) diff = -diff;
            if(diff > THRESHOLD) {
                numErrors++;
            }
        }
    }
clean:
    if (hostIfromdev) free(hostIfromdev);
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

template <typename T_ELEM>
int doWgrad(
    cudnnHandle_t handle_,
    T_ELEM* devPtrI,
    T_ELEM* devPtrF,
    T_ELEM* devPtrO,
    T_ELEM* hostI,
    T_ELEM* hostF,
    T_ELEM* hostO,
    cudnnTensorDescriptor_t cudnnIdesc,
    cudnnFilterDescriptor_t   cudnnFdesc,
    cudnnTensorDescriptor_t cudnnOdesc,
    cudnnConvolutionDescriptor_t cudnnConvDesc,
    float alpha,
    float beta,
    cudnnTensorFormat_t filterFormat,
    const int*   dimA,
    const int*   filterdimA,
    const int*   outdimA,
    const int*   strideA,
    const int*   outstrideA,
    const int*   convstrideA,
    const int*   padA,
    const int*   dilationA,
    const int   benchmark) {

    int filsize = filterdimA[0]*filterdimA[1]*filterdimA[2]*filterdimA[3];
    T_ELEM* hostFfromdev = (T_ELEM*)calloc (filsize, sizeof(hostF[0]) );
    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    void *workSpace = 0;
    size_t workSpaceSize;
    int numErrors = 0;
    double start, stop;

    checkCudnnErr ( cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_, cudnnIdesc, cudnnOdesc, cudnnConvDesc,
        cudnnFdesc, algo, &workSpaceSize) );

    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    start = second();
    checkCudnnErr ( cudnnConvolutionBackwardFilter (handle_,
                                                     (void*)(&alpha),
                                                     cudnnIdesc, devPtrI,
                                                     cudnnOdesc, devPtrO,
                                                     cudnnConvDesc,
                                                     algo,
                                                     workSpace, workSpaceSize,
                                                     (void*)(&beta),
                                                     cudnnFdesc, devPtrF) );
    checkCudaErr( cudaDeviceSynchronize() );
    stop = second();
    printPerf( stop - start, 0, 0,
                      0, 0, 0, 0);
    checkCudaErr( cudaMemcpy(hostFfromdev, devPtrF, sizeof(hostF[0]) * filsize, cudaMemcpyDeviceToHost) );
    checkCudaErr( cudaDeviceSynchronize() );

    if (!benchmark) {
        weightGrad_cpu_ref<T_ELEM>(hostI, hostO, alpha, beta, hostF, filterFormat, dimA, filterdimA, outdimA, strideA, outstrideA, convstrideA, padA, dilationA, 4);
        for (int index = 0; index < filsize; index++) { // assuming in data is packed
            float diff = getError(hostFfromdev[index], hostF[index]);
            if (diff < 0) diff = -diff;
            if(diff > THRESHOLD) {
                numErrors++;
            }
        }
    }
clean:
    if (hostFfromdev) free(hostFfromdev);
    if (workSpace) cudaFree(workSpace);
    return numErrors;
}

template <typename T_ELEM>
int doTest(int algo, int* dimA, int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType, int mathType, int benchmark, bool fold, cudnnConvolutionMode_t mode, cudnnReorderType_t reorder) {

    cudnnHandle_t handle_;
    T_ELEM* devPtrI = NULL;
    T_ELEM* devPtrF = NULL;
    T_ELEM* devPtrReorderedF = NULL;
    float* devPtrBias = NULL;
    float* devPtrReorderedBias = NULL;
    T_ELEM* devPtrZ = NULL;
    T_ELEM* devPtrO = NULL;
    T_ELEM* hostI = NULL;
    T_ELEM* hostF = NULL;
    T_ELEM* hostO = NULL;
    T_ELEM* hostZ = NULL;
    float* hostBias = NULL;
    float* hostReorderedBias = NULL;

    cudnnTensorDescriptor_t cudnnIdesc;
    cudnnFilterDescriptor_t cudnnFdesc;
    cudnnTensorDescriptor_t cudnnOdesc;
    cudnnTensorDescriptor_t cudnnBiasdesc;
	cudnnActivationDescriptor_t activationDesc;
    cudnnConvolutionDescriptor_t cudnnConvDesc;
    
    bool convBiasAct = (algo == 1) ? true : false;

    int convDim = 2;

    float alpha     = 0.8f;
    float beta      = 0.0f;
    int numErrors   = 0;
    int dilationA[] = {1, 1};
    int insize      = 0;
    int filtersize  = 0;
    int outdimA[]   = {1, 8, 30, 30};
    int biasdimA[4];
    int biasstrideA[4];
    int outsize  = 0;
    int biasSize = 1;

    int dimA_padded[4];
    int outdimA_padded[4];
    int filterdimA_padded[4];
    int biasdimA_padded[4];
    int strideA_padded[4];
    int outstrideA_padded[4];
    int filterstrideA_padded[4];
    int biasstrideA_padded[4];


    outdimA[0] = dimA[0];
    outdimA[1] = filterdimA[0];
    for( int dim = 0; dim < 2; dim++) {
        outdimA[dim+2] = getFwdConvOutputDim( dimA[dim+2],
                                              padA[dim],
                                              filterdimA[dim+2],
                                              convstrideA[dim],
                                              dilationA[dim]);
    }

    for (int i = 0; i < 4; i++) {
        dimA_padded[i] = dimA[i];
        outdimA_padded[i] = outdimA[i];
        filterdimA_padded[i] = filterdimA[i];
    }

    // If vectorized, pad to proper dims. If not vectorized. dim_padded == dim
    if (filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        int vecSize;
        if (dataType == CUDNN_DATA_INT8x32) {
            vecSize = 32;
        } else {
            vecSize = 4;
        }

        if (dimA_padded[1] % vecSize != 0) {
            dimA_padded[1] = dimA_padded[1] + vecSize - ((int8_t)dimA_padded[1] % vecSize);
        }
        if (filterdimA_padded[1] % vecSize != 0) {
            filterdimA_padded[1] = filterdimA_padded[1] + vecSize - ((int8_t)filterdimA_padded[1] % vecSize);
        }
        if (filterdimA_padded[0] % vecSize != 0) {
            filterdimA_padded[0] = filterdimA_padded[0] + vecSize - ((int8_t)filterdimA_padded[0] % vecSize);
            outdimA_padded[1] = filterdimA_padded[0];
        }
    }

    printf("====USER DIMENSIONS====\n");
    printf("input dims are %d, %d, %d, %d\n", dimA[0], dimA[1], dimA[2], dimA[3]);
    printf("filter dims are %d, %d, %d, %d\n", filterdimA[0], filterdimA[1], filterdimA[2], filterdimA[3]);
    printf("output dims are %d, %d, %d, %d\n", outdimA[0], outdimA[1], outdimA[2], outdimA[3]);
    printf("====PADDING DIMENSIONS====\n");
    printf("padded input dims are %d, %d, %d, %d\n", dimA_padded[0], dimA_padded[1], dimA_padded[2], dimA_padded[3]);
    printf("padded filter dims are %d, %d, %d, %d\n", filterdimA_padded[0], filterdimA_padded[1], filterdimA_padded[2], filterdimA_padded[3]);
    printf("padded output dims are %d, %d, %d, %d\n", outdimA_padded[0], outdimA_padded[1], outdimA_padded[2], outdimA_padded[3]);

    checkCudnnErr(cudnnCreate(&handle_));
    
    checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnIdesc ));
    checkCudnnErr( cudnnCreateFilterDescriptor( &cudnnFdesc ));
    checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnOdesc ));
    checkCudnnErr( cudnnCreateTensorDescriptor( &cudnnBiasdesc ));
    checkCudnnErr( cudnnCreateActivationDescriptor(&activationDesc));
    checkCudnnErr( cudnnCreateConvolutionDescriptor( &cudnnConvDesc ));

    generateStrides(dimA_padded, strideA_padded, 4, filterFormat);
    insize = dimA_padded[0] * dimA_padded[1] * dimA_padded[2] * dimA_padded[3];

    generateStrides(filterdimA_padded, filterstrideA_padded, 4, filterFormat);
    filtersize = filterdimA_padded[0] * filterdimA_padded[1] * filterdimA_padded[2] * filterdimA_padded[3];

    generateStrides(outdimA_padded, outstrideA_padded, 4, filterFormat);
    outsize = outdimA_padded[0] * outdimA_padded[1] * outdimA_padded[2] * outdimA_padded[3];
    
    for (int i = 0; i < 4; i++) {
        biasdimA[i]           = 1;
        biasdimA_padded[i]    = 1;
        biasstrideA[i] 		  = 1;
        biasstrideA_padded[i] = 1;
    }

    biasdimA[1]           = outdimA[1];
    biasdimA_padded[1]    = outdimA_padded[1];
    biasstrideA[0]        = biasdimA[1];
    biasstrideA_padded[0] = biasdimA_padded[1];

    for (int i = 0; i < 4; i++) {
        biasSize *= biasdimA_padded[i];
    }
    
    cudaMalloc((void**)&(devPtrI), (insize) * sizeof(devPtrI[0]));
    cudaMalloc((void**)&(devPtrF), (filtersize) * sizeof(devPtrF[0]));
    cudaMalloc((void**)&(devPtrReorderedF), (filtersize) * sizeof(devPtrF[0]));
    cudaMalloc((void**)&(devPtrO), (outsize) * sizeof(devPtrO[0]));
    cudaMalloc((void**)&(devPtrZ), (outsize) * sizeof(devPtrZ[0]));
    cudaMalloc((void**)&(devPtrBias), (biasSize) * sizeof(devPtrBias[0]));
    cudaMalloc((void**)&(devPtrReorderedBias), (biasSize) * sizeof(devPtrBias[0]));
    hostI             = (T_ELEM*)calloc(insize, sizeof(hostI[0]));
    hostF             = (T_ELEM*)calloc(filtersize, sizeof(hostF[0]));
    hostO             = (T_ELEM*)calloc(outsize, sizeof(hostO[0]));
    hostZ             = (T_ELEM*)calloc(outsize, sizeof(hostZ[0]));
    hostBias          = (float*)calloc(biasSize, sizeof(hostBias[0]));
    hostReorderedBias = (float*)calloc(biasSize, sizeof(hostReorderedBias[0]));
    
	if (filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        initImagePadded((int8_t*)hostI, dimA, dimA_padded, strideA_padded, dataType);
        initImagePadded((int8_t*)hostF, filterdimA, filterdimA_padded, filterstrideA_padded, dataType);
        initImagePadded((int8_t*)hostO, outdimA, outdimA_padded, outstrideA_padded, dataType);
        initImagePadded((int8_t*)hostZ, outdimA, outdimA_padded, outstrideA_padded, dataType);
    } else {
        initImage(hostI, insize);
        initImage(hostF, filtersize);
        initImage(hostO, outsize);
        initImage(hostZ, outsize);       
    }

    checkCudaErr(cudaMemcpy(devPtrI, hostI, sizeof(hostI[0]) * insize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrF, hostF, sizeof(hostF[0]) * filtersize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrO, hostO, sizeof(hostO[0]) * outsize, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devPtrZ, hostZ, sizeof(hostZ[0]) * outsize, cudaMemcpyHostToDevice));
    if(convBiasAct){
		initImage(hostBias, biasSize);
        initImage(hostReorderedBias, biasSize);
		checkCudaErr(cudaMemcpy(devPtrBias, hostBias, sizeof(hostBias[0]) * biasSize, cudaMemcpyHostToDevice));
	}
    checkCudaErr(cudaDeviceSynchronize());

    //Dimensions are different for INT8x4 and INT8x32, so let function decide rather than using our standard dimensions
    if (filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
        checkCudnnErr( cudnnSetTensorNdDescriptorEx(cudnnIdesc, filterFormat, dataType, convDim+2, dimA_padded) );
        checkCudnnErr( cudnnSetTensorNdDescriptorEx(cudnnOdesc, filterFormat, dataType, convDim+2, outdimA_padded) );
        checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                       convDim,
                                                       padA,
                                                       convstrideA,
                                                       dilationA,
                                                       mode,
                                                       CUDNN_DATA_INT32) );
    } else {
        checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnIdesc, dataType, convDim+2, dimA_padded, strideA_padded) );
        checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnOdesc, dataType, convDim+2, outdimA_padded, outstrideA_padded) );
        checkCudnnErr( cudnnSetConvolutionNdDescriptor(cudnnConvDesc,
                                                       convDim,
                                                       padA,
                                                       convstrideA,
                                                       dilationA,
                                                       mode,
                                                       CUDNN_DATA_FLOAT) );
    }
    checkCudnnErr( cudnnSetFilterNdDescriptor(cudnnFdesc, dataType, filterFormat, convDim+2, filterdimA_padded));
    if(convBiasAct){
		checkCudnnErr( cudnnSetTensorNdDescriptor(cudnnBiasdesc, CUDNN_DATA_FLOAT, convDim + 2, biasdimA_padded, biasstrideA_padded));
		checkCudnnErr( cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, /*reluCeiling*/ 10000.0));
	}
    
    // The below avoids extra if statements for simplicity; if the use case doesn't need to reorder the filter then the user needs not use devPrtReorderedF at all
    if(reorder == CUDNN_DEFAULT_REORDER){
			if(dataType == CUDNN_DATA_INT8x32){
				checkCudnnErr( cudnnReorderFilterAndBias(handle_, 
														 cudnnFdesc, 
														 CUDNN_DEFAULT_REORDER, 
														 devPtrF, 
														 devPtrReorderedF, 
														 convBiasAct, 
														 devPtrBias, 
														 devPtrReorderedBias));
														 
				checkCudnnErr( cudnnSetConvolutionReorderType(cudnnConvDesc, CUDNN_NO_REORDER)); 
			}else{
				printf("ERROR: incorrect parameters provided for filter and bias reorder\n");
				return -1;
			}
	}else{
		checkCudaErr( cudaMemcpy(devPtrReorderedF, devPtrF, sizeof(devPtrF[0]) * filtersize, cudaMemcpyDeviceToDevice));
		if(convBiasAct){
			checkCudaErr( cudaMemcpy(devPtrReorderedBias, devPtrBias, sizeof(devPtrBias[0]) * biasSize, cudaMemcpyDeviceToDevice));
		}
	}
			
    if (mathType == 1) {
        checkCudnnErr( cudnnSetConvolutionMathType(cudnnConvDesc, CUDNN_TENSOR_OP_MATH) );
    }

    if (algo == 0) {
        printf("Testing conv\n");
        numErrors = doConv(
            handle_,
            devPtrI,
            devPtrReorderedF,
            devPtrO,
            hostI,
            hostF,
            hostO,
            cudnnIdesc,
            cudnnFdesc,
            cudnnOdesc,
            cudnnConvDesc,
            alpha,
            beta,
            filterFormat,
            dataType,
            dimA_padded,
            filterdimA_padded,
            outdimA_padded,
            strideA_padded,
            outstrideA_padded,
            convstrideA,
            padA,
            dilationA,
            benchmark);
    } else if (algo == 1) {
		printf("Testing convBiasAct\n");
        numErrors = doConvBiasAct(
			  handle_,
			  devPtrI,
			  devPtrReorderedF,
			  devPtrO,
			  devPtrReorderedBias,
			  devPtrZ,
			  hostI,
			  hostF,
			  hostO,
			  hostBias,
			  hostZ,
			  cudnnIdesc,
			  cudnnFdesc,
			  cudnnOdesc,
			  cudnnBiasdesc,
			  activationDesc,
			  cudnnConvDesc,
			  alpha,
			  beta,
			  filterFormat,
			  dataType,
			  dimA_padded,
			  filterdimA_padded,
			  outdimA_padded,
			  biasdimA_padded,
			  strideA_padded,
			  outstrideA_padded,
			  biasstrideA_padded,
			  convstrideA,
			  padA,
			  dilationA,
			  benchmark);
    } else if (algo == 2) {
        printf("Testing dgrad\n");
        numErrors = doDgrad(
            handle_,
            devPtrI,
            devPtrF,
            devPtrO,
            hostI,
            hostF,
            hostO,
            cudnnIdesc,
            cudnnFdesc,
            cudnnOdesc,
            cudnnConvDesc,
            alpha,
            beta,
            filterFormat,
            dimA,
            filterdimA,
            outdimA,
            strideA_padded,
            outstrideA_padded,
            convstrideA,
            padA,
            dilationA,
            benchmark,
            fold,
            mode);
    } else {
        printf("Testing wgrad\n");
        numErrors = doWgrad(
            handle_,
            devPtrI,
            devPtrF,
            devPtrO,
            hostI,
            hostF,
            hostO,
            cudnnIdesc,
            cudnnFdesc,
            cudnnOdesc,
            cudnnConvDesc,
            alpha,
            beta,
            filterFormat,
            dimA,
            filterdimA,
            outdimA,
            strideA_padded,
            outstrideA_padded,
            convstrideA,
            padA,
            dilationA,
            benchmark);
    }

	if(!benchmark){
		if (numErrors == 0) {
			printf("Test PASSED\n");
		} else {
			printf("Test FAILED, num errors = %d\n", numErrors);
		}
	}

clean:
    if (devPtrI) cudaFree (devPtrI);
    if (devPtrReorderedF) cudaFree(devPtrReorderedF);
    if (devPtrF) cudaFree (devPtrF);
    if (devPtrO) cudaFree (devPtrO);
    if (devPtrZ) cudaFree (devPtrZ);
    if (devPtrBias) cudaFree (devPtrBias);
    if (devPtrReorderedBias) cudaFree (devPtrReorderedBias);
    if (hostI) free(hostI);
    if (hostF) free(hostF);
    if (hostO) free(hostO);
    if (hostZ) free(hostZ);
    if (hostBias) free(hostBias);
    if (hostReorderedBias) free(hostReorderedBias);
    if (cudnnIdesc) cudnnDestroyTensorDescriptor(cudnnIdesc);
    if (cudnnFdesc) cudnnDestroyFilterDescriptor(cudnnFdesc);
    if (cudnnOdesc) cudnnDestroyTensorDescriptor(cudnnOdesc);
    if (cudnnBiasdesc && convBiasAct) cudnnDestroyTensorDescriptor(cudnnBiasdesc);
    if (activationDesc && convBiasAct) cudnnDestroyActivationDescriptor(activationDesc);
    if (cudnnConvDesc) cudnnDestroyConvolutionDescriptor(cudnnConvDesc);
    if (handle_) cudnnDestroy(handle_);

    return 0;
}

inline void selectDataType(int userSpecifiedDataType, cudnnDataType_t &dataType, int &error)
{
	switch(userSpecifiedDataType){
		case 0: 
			dataType = CUDNN_DATA_FLOAT;
			break;
		case 1: 
			dataType = CUDNN_DATA_HALF;
			break;
		case 2: 
			dataType = CUDNN_DATA_INT8x4;
			break;
		case 3: 
			dataType = CUDNN_DATA_INT8x32;
			break;
		default: 
			error++;
			break;
	}
}

int main( int argc, char** argv )
{
    int algo = 0;
    int mathType = 0;
    int benchmark = 0;
    int reorderMode = 0;
    int userSpecifiedDataType = 0;
	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int dimA[] = {1, 32, 4, 4};

    int padA[] = {0, 0};
    int convstrideA[] = {1, 1};

    //batch size and feature layers must be multiples of 4 or 32 when using int8x4 or int8x32 respectively
    int filterdimA[] = {32, 32, 1, 1};

    cudnnTensorFormat_t  filterFormat = CUDNN_TENSOR_NCHW;
    bool fold = false;
    cudnnConvolutionMode_t mode = CUDNN_CONVOLUTION;
    cudnnReorderType_t reorder = CUDNN_NO_REORDER;

    int error = 0;
    argc -= 1;
    argv++;
    while (argc) {
        if (*argv[0] == SWITCH_CHAR) {
            switch (*(argv[0]+1)) {
            case 'b':
                benchmark = 1;
                break;
            case 'c':
                if (strncmp(argv[0] + 1, "convBiasAct", strlen("convBiasAct")) == 0) {
					algo = 1;
                } else {
					dimA[1]       = atol(argv[0] + 2);
					filterdimA[1] = dimA[1];
                }
                break;
            case 'd':
                if ( strncmp( argv[0]+1, "dgrad" , strlen("dgrad")) == 0) {
                     algo = 2;
                }else if ( strncmp( argv[0]+1, "dataType" , strlen("dataType")) == 0) {
                     userSpecifiedDataType = (cudnnDataType_t)(atoi(argv[0]+ 1 + strlen("dataType")));
                     selectDataType(userSpecifiedDataType, dataType, error);
                }
                break;
            case 'f':
                if ( strncmp( argv[0]+1, "filterFormat" , strlen("filterFormat")) == 0) {
                     filterFormat = (cudnnTensorFormat_t)(atoi(argv[0]+ 1 + strlen("filterFormat")));
                }
                if ( strncmp( argv[0]+1, "fold" , strlen("fold")) == 0) {
                     fold = true;
                }
                
                break;
            case 'h':
                dimA[2] = atol(argv[0]+2);
                break;
            case 'k':
                filterdimA[0] = atol(argv[0]+2);
                break;
            case 'm':
                if ( strncmp( argv[0]+1, "mathType1" , strlen("mathType1")) == 0) {
                     mathType = 1;
                }
                break;
     	    case 'n':
                dimA[0] = atol(argv[0]+2);
                break;
            case 'p':
                if ( strncmp( argv[0]+1, "pad_h" , strlen("pad_h")) == 0) {
                    padA[0] = (int)atol(argv[0]+ 1 + strlen("pad_h"));
                }
                else if ( strncmp( argv[0]+1, "pad_w" , strlen("pad_w")) == 0) {
                    padA[1] = (int)atol(argv[0]+ 1 + strlen("pad_w"));
                }
                break;
            case 'r':
                if ( strncmp( argv[0]+1, "reorderMode" , strlen("reorderMode")) == 0) {
                    reorderMode = (int)atoi(argv[0]+ 1 + strlen("reorderMode"));
                    reorder = (reorderMode == 0) ? CUDNN_NO_REORDER : CUDNN_DEFAULT_REORDER;
                }else{
					filterdimA[2] = atol(argv[0]+2);
				}
                break;
            case 's':
                filterdimA[3] = atol(argv[0]+2);
                break;
            case 'u':
                convstrideA[0] = atol(argv[0]+2);
                break;
            case 'v':
                convstrideA[1] = atol(argv[0]+2);
                break;
            case 'w':
                if ( strncmp( argv[0]+1, "wgrad" , strlen("wgrad")) == 0) {
                     algo = 3;
                }
                else dimA[3] = atol(argv[0]+2);
                break;
            case 'x':
                mode = CUDNN_CROSS_CORRELATION;
                break;
            default:
                error++;
                break;
            }
            if (error) {
                fprintf(stderr, "Unknown switch '%c%s'\n", SWITCH_CHAR, argv[0]+1);
                return error;
            }
        }
        else {
            fprintf(stderr, "Invalid separator '%c' for option '%s'\n", *argv[0], argv[0] );
            return 1;
        }
        argc -= 1;
        argv++;
    }

    int device;
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    int deviceVer = devProp.major * 10 + devProp.minor;
    
    if(dataType == CUDNN_DATA_FLOAT){
		if(filterFormat == CUDNN_TENSOR_NCHW_VECT_C){
			printf("INT8x4 and INT8x32 tests use CUDNN_TENSOR_NCHW_VECT_C\n");
			return -1;
		}
	} else if(dataType == CUDNN_DATA_HALF){
		if(filterFormat == CUDNN_TENSOR_NCHW_VECT_C){
			printf("INT8x4 and INT8x32 tests use CUDNN_TENSOR_NCHW_VECT_C\n");
			return -1;
		}
	} else if(dataType == CUDNN_DATA_INT8x4){
		if(filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NHWC){
			printf("For single and double precision tests use a different format\n");
			return -1;
		}else if(deviceVer < 61){
			printf("WAIVED: Device version %d does not support int8x4!\n", deviceVer);
			return 0;
		} else if(filterdimA[0]%32 != 0){
			printf("Filter output feature map count ('k = %d') is not a multiple of 32\n", filterdimA[0]);
			return -1;
		} else if(filterdimA[1]%32 != 0){
			printf("Filter input feature map count ('c = %d') is not a multiple of 32\n", filterdimA[1]);
			return -1;
		}
	} else if(dataType == CUDNN_DATA_INT8x32){
		if(filterFormat == CUDNN_TENSOR_NCHW || filterFormat == CUDNN_TENSOR_NHWC){
			printf("For single and double precision tests use a different format\n");
			return -1;
		}else if(deviceVer < 72){
			printf("WAIVED: Device version %d does not support int8x32!\n", deviceVer);
			return 0;
		} else if(filterdimA[0]%32 != 0){
			printf("Filter output feature map count ('k = %d') is not a multiple of 32\n", filterdimA[0]);
			return -1;
		} else if(filterdimA[1]%32 != 0){
			printf("Filter input feature map count ('c = %d') is not a multiple of 32\n", filterdimA[1]);
			return -1;
		}
	}else{
		printf("ERROR: This sample only supports FLOAT, HALF, INT8x4 and INT8x32 DATATYPE!\n");
		return -1;
	}
	
	
	typedef int (*doTest_launcher)(int algo, int* dimA, int* padA, int* convstrideA, int* filterdimA, cudnnTensorFormat_t filterFormat, cudnnDataType_t dataType, int mathType, int benchmark, bool fold, cudnnConvolutionMode_t mode, cudnnReorderType_t reorder);
	
	const doTest_launcher doTest_funcs[3] = {
		doTest<float>,
		doTest<half1>,
		doTest<int8_t>
	};	
	userSpecifiedDataType = (userSpecifiedDataType == 3) ? 2 : userSpecifiedDataType;
	doTest_funcs[userSpecifiedDataType](algo, dimA, padA, convstrideA, filterdimA, filterFormat, dataType, mathType, benchmark, fold, mode, reorder);

    return 0;
}
