/**
* Copyright 2016 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cudnn.h>
#include <cuda.h>
#include <stdio.h>
#include "fp16_emu.h"

// Usage
//   > ./RNN <flags>
// Command line flags
//   -seqLength<int>    : Specify sequence length
//   -numLayers<int>    : Specify number of layers
//   -hiddenSize<int>   : Specify hidden size
//   -miniBatch<int>    : Specify minibatch size
//   -dropout<float>    : Specify dropout probability
//   -bidirectional     : Switch to bidirectional instead of unidirectional RNN
//   -mode{0,1,2,3}     : Specify mode (ReLU, tanh, LSTM, GRU)
//   -persistent{0,1,2} : Specify recurrence algorithm (standard, persist dynamic, persist static)
//   -P{s,d,h}          : Specify data type precision (float, double, half)
//   -H                 : Display this help message
//
// Reference outputs (calculated on an M40 GPU)
// golden_1.txt (default case if you just run ./RNN)
// > ./RNN -seqLength20 -numLayers2 -hiddenSize512 -inputSize512 -miniBatch64 -mode0
// Forward: 1250 GFLOPS
// Backward: 1896 GFLOPS, (1299 GFLOPS), (3511 GFLOPS)
// y checksum 1.315793E+06     hy checksum 1.315212E+05
// dx checksum 6.676003E+01    dhx checksum 6.425050E+01
// dw checksum 1.453750E+09
//
// golden_2.txt
// > ./RNN -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -miniBatch64 -mode1
// Forward: 1225 GFLOPS
// Backward: 1910 GFLOPS, (1299 GFLOPS), (3601 GFLOPS)
// y checksum 6.319591E+05     hy checksum 6.319605E+04
// dx checksum 4.501830E+00    dhx checksum 4.489543E+00
// dw checksum 5.012598E+07
//
// golden_3.txt
// > ./RNN -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -miniBatch64 -mode2
// Forward: 2569 GFLOPS
// Backward: 2654 GFLOPS, (2071 GFLOPS), (3694 GFLOPS)
// y checksum 5.749536E+05     cy checksum 4.365091E+05     hy checksum 5.774818E+04
// dx checksum 3.842206E+02    dcx checksum 9.323785E+03    dhx checksum 1.182562E+01
// dw checksum 4.313461E+08
//
// golden_4.txt
// > ./RNN -seqLength20 -numLayers2 -inputSize512 -hiddenSize512 -miniBatch64 -mode3
// Forward: 2310 GFLOPS
// Backward: 2536 GFLOPS, (1955 GFLOPS), (3606 GFLOPS)
// y checksum 6.358978E+05     hy checksum 6.281680E+04
// dx checksum 6.296622E+00    dhx checksum 2.289960E+05
// dw checksum 5.397419E+07

// Templated functions to get cudnnDataType_t from a templated type
template <typename T_ELEM> __inline__ cudnnDataType_t getDataType();
template <> __inline__ cudnnDataType_t getDataType<double>() { return CUDNN_DATA_DOUBLE; }
template <> __inline__ cudnnDataType_t getDataType<float>()  { return CUDNN_DATA_FLOAT;  }
template <> __inline__ cudnnDataType_t getDataType<half1>()  { return CUDNN_DATA_HALF;   }

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cudnnErrCheck(stat) { cudnnErrCheck_((stat), __FILE__, __LINE__); }
void cudnnErrCheck_(cudnnStatus_t stat, const char *file, int line) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
    }
}

// Kernel and launcher to initialize GPU data to some constant value
template <typename T_ELEM>
__global__
void initGPUData_ker(T_ELEM *data, int numElements, T_ELEM value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numElements) {
        data[tid] = value;
    }
}

template <typename T_ELEM>
void initGPUData(T_ELEM *data, int numElements, T_ELEM value) {
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 1024;
    gridDim.x  = (numElements + blockDim.x - 1) / blockDim.x;

    initGPUData_ker<<<gridDim, blockDim>>>(data, numElements, value);
}

// This function does all the work of setting up and running cuDNN's RNN functions with the given parameters.
// It also calculates performance results and checksums, printing them to the command line and saving them to
// result.txt for potential comparison to the files (golden_1.txt, etc).
template <typename T_ELEM>
void doTest(int seqLength,
            int numLayers,
            int hiddenSize,
            int inputSize,
            int miniBatch,
            float dropout,
            bool bidirectional,
            cudnnRNNMode_t mode,
            cudnnRNNAlgo_t algo) {
    FILE *fp;
    fp = fopen("result.txt", "w");

    // -------------------------
    // Create cudnn context
    // -------------------------
    cudnnHandle_t cudnnHandle;
    cudnnErrCheck(cudnnCreate(&cudnnHandle));

    // -------------------------
    // Set up inputs and outputs
    // -------------------------
    void *x;
    void *hx = NULL;
    void *cx = NULL;

    void *dx;
    void *dhx = NULL;
    void *dcx = NULL;

    void *y;
    void *hy = NULL;
    void *cy = NULL;

    void *dy;
    void *dhy = NULL;
    void *dcy = NULL;

    int dimIn[3];
    int dimOut[3];
    int dimHidden[3];
    int strideIn[3];
    int strideOut[3];
    int strideHidden[3];

    // Set up required dimensions for input, output, and hidden state tensors
    dimIn[0] = miniBatch;
    dimIn[1] = inputSize;
    dimIn[2] = 1;
    dimOut[0] = miniBatch;
    dimOut[1] = hiddenSize * (bidirectional ? 2 : 1);
    dimOut[2] = 1;
    dimHidden[0] = numLayers * (bidirectional ? 2 : 1);
    dimHidden[1] = miniBatch;
    dimHidden[2] = hiddenSize;

    strideIn[0] = dimIn[1] * dimIn[2];
    strideIn[1] = dimIn[2];
    strideIn[2] = 1;
    strideOut[0] = dimOut[1] * dimOut[2];
    strideOut[1] = dimOut[2];
    strideOut[2] = 1;
    strideHidden[0] = dimHidden[1] * dimHidden[2];
    strideHidden[1] = dimHidden[2];
    strideHidden[2] = 1;

    // Calculating total elements per each
    int inputTensorSize  = dimIn[0] * dimIn[1] * dimIn[2];
    int outputTensorSize = dimOut[0] * dimOut[1] * dimOut[2];
    int hiddenTensorSize = dimHidden[0] * dimHidden[1] * dimHidden[2];

    // Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
    cudaErrCheck(cudaMalloc((void **)&x, seqLength * inputTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&y, seqLength * outputTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dx, seqLength * inputTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dy, seqLength * outputTensorSize * sizeof(T_ELEM)));

    cudaErrCheck(cudaMalloc((void **)&hx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&cx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&hy, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&cy, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dhx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dcx, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dhy, hiddenTensorSize * sizeof(T_ELEM)));
    cudaErrCheck(cudaMalloc((void **)&dcy, hiddenTensorSize * sizeof(T_ELEM)));

    // Set up tensor descriptors. x/y/dx/dy are arrays, one per time step.
    cudnnTensorDescriptor_t *xDesc, *yDesc, *dxDesc, *dyDesc;
    cudnnTensorDescriptor_t hxDesc, cxDesc;
    cudnnTensorDescriptor_t hyDesc, cyDesc;
    cudnnTensorDescriptor_t dhxDesc, dcxDesc;
    cudnnTensorDescriptor_t dhyDesc, dcyDesc;

    xDesc  = (cudnnTensorDescriptor_t *)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    yDesc  = (cudnnTensorDescriptor_t *)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    dxDesc = (cudnnTensorDescriptor_t *)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));
    dyDesc = (cudnnTensorDescriptor_t *)malloc(seqLength * sizeof(cudnnTensorDescriptor_t));

    // In this example dimA[1] is constant across the whole sequence
    // This isn't required, all that is required is that it does not increase.
    for (int i = 0; i < seqLength; i++) {
        cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));

        cudnnErrCheck(cudnnCreateTensorDescriptor(&dxDesc[i]));
        cudnnErrCheck(cudnnCreateTensorDescriptor(&dyDesc[i]));

        cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], getDataType<T_ELEM>(), 3, dimIn, strideIn));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dxDesc[i], getDataType<T_ELEM>(), 3, dimIn, strideIn));

        cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], getDataType<T_ELEM>(), 3, dimOut, strideOut));
        cudnnErrCheck(cudnnSetTensorNdDescriptor(dyDesc[i], getDataType<T_ELEM>(), 3, dimOut, strideOut));
    }

    cudnnErrCheck(cudnnCreateTensorDescriptor(&hxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&hyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&cyDesc));

    cudnnErrCheck(cudnnCreateTensorDescriptor(&dhxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dcxDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dhyDesc));
    cudnnErrCheck(cudnnCreateTensorDescriptor(&dcyDesc));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(hxDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cxDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(hyDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(cyDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));

    cudnnErrCheck(cudnnSetTensorNdDescriptor(dhxDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dcxDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dhyDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));
    cudnnErrCheck(cudnnSetTensorNdDescriptor(dcyDesc, getDataType<T_ELEM>(), 3, dimHidden, strideHidden));

    // -------------------------
    // Set up the dropout descriptor (needed for the RNN descriptor)
    // -------------------------
    unsigned long long seed = 1337ull;  // Pick a seed.

    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));

    // How much memory does dropout need for states?
    // These states are used to generate random numbers internally
    // and should not be freed until the RNN descriptor is no longer used
    size_t stateSize;
    void *states;
    cudnnErrCheck(cudnnDropoutGetStatesSize(cudnnHandle, &stateSize));

    cudaErrCheck(cudaMalloc(&states, stateSize));

    cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc,
                                            cudnnHandle,
                                            dropout,
                                            states,
                                            stateSize,
                                            seed));

    // -------------------------
    // Set up the RNN descriptor
    // -------------------------
    cudnnRNNDescriptor_t rnnDesc;

    cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));

    cudnnErrCheck(cudnnSetRNNDescriptor_v6(cudnnHandle,
                                           rnnDesc,
                                           hiddenSize,
                                           numLayers,
                                           dropoutDesc,
                                           CUDNN_LINEAR_INPUT,  // We can also skip the input matrix transformation
                                           bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
                                           mode,
                                           algo,  // Can be changed to use persistent RNNs on Pascal+ GPUs.
                                           getDataType<T_ELEM>()));

    // -------------------------
    // Set up parameters
    // -------------------------
    // This needs to be done after the rnn descriptor is set as otherwise
    // we don't know how many parameters we have to allocate
    void *w;
    void *dw;

    cudnnFilterDescriptor_t wDesc, dwDesc;

    cudnnErrCheck(cudnnCreateFilterDescriptor(&wDesc));
    cudnnErrCheck(cudnnCreateFilterDescriptor(&dwDesc));

    size_t weightsSize;
    cudnnErrCheck(cudnnGetRNNParamsSize(cudnnHandle, rnnDesc, xDesc[0], &weightsSize, getDataType<T_ELEM>()));

    int dimW[3];
    dimW[0] = weightsSize / sizeof(T_ELEM);
    dimW[1] = 1;
    dimW[2] = 1;

    cudnnErrCheck(cudnnSetFilterNdDescriptor(wDesc, getDataType<T_ELEM>(), CUDNN_TENSOR_NCHW, 3, dimW));
    cudnnErrCheck(cudnnSetFilterNdDescriptor(dwDesc, getDataType<T_ELEM>(), CUDNN_TENSOR_NCHW, 3, dimW));

    cudaErrCheck(cudaMalloc((void **)&w, weightsSize));
    cudaErrCheck(cudaMalloc((void **)&dw, weightsSize));

    // -------------------------
    // Set up work space and reserved memory
    // -------------------------
    void *workspace;
    void *reserveSpace;

    size_t workSize;
    size_t reserveSize;

    // Need for every pass
    cudnnErrCheck(cudnnGetRNNWorkspaceSize(cudnnHandle, rnnDesc, seqLength, xDesc, &workSize));
    // Only needed in training, shouldn't be touched between passes.
    cudnnErrCheck(cudnnGetRNNTrainingReserveSize(cudnnHandle, rnnDesc, seqLength, xDesc, &reserveSize));

    cudaErrCheck(cudaMalloc((void **)&workspace, workSize));
    cudaErrCheck(cudaMalloc((void **)&reserveSpace, reserveSize));

    // *********************************************************************************************************
    // Initialise weights and inputs
    // *********************************************************************************************************
    // We initialise to something simple.
    // Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.

    //Initialize inputs
    initGPUData<T_ELEM>((T_ELEM *)x, seqLength * inputTensorSize, 1.0);
    if (hx != NULL) initGPUData<T_ELEM>((T_ELEM *)hx, hiddenTensorSize, 1.0);
    if (cx != NULL) initGPUData<T_ELEM>((T_ELEM *)cx, hiddenTensorSize, 1.0);

    initGPUData<T_ELEM>((T_ELEM *)dy, seqLength * outputTensorSize, 1.0);
    if (dhy != NULL) initGPUData<T_ELEM>((T_ELEM *)dhy, hiddenTensorSize, 1.0);
    if (dcy != NULL) initGPUData<T_ELEM>((T_ELEM *)dcy, hiddenTensorSize, 1.0);


    // Initialize Weights
    int numLinearLayers = 0;
    if (mode == CUDNN_RNN_RELU || mode == CUDNN_RNN_TANH) {
        numLinearLayers = 2;
    } else if (mode == CUDNN_LSTM) {
        numLinearLayers = 8;
    } else if (mode == CUDNN_GRU) {
        numLinearLayers = 6;
    }

    for (int layer = 0; layer < numLayers * (bidirectional ? 2 : 1); layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];

            //Initialize layer weights
            cudnnFilterDescriptor_t linLayerMatDesc;
            T_ELEM *linLayerMat;

            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
            cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(cudnnHandle,
                                                          rnnDesc,
                                                          layer,
                                                          xDesc[0],
                                                          wDesc,
                                                          w,
                                                          linLayerID,
                                                          linLayerMatDesc,
                                                          (void **)&linLayerMat));

            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format, &nbDims, filterDimA));

            initGPUData<T_ELEM>(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.0 / (filterDimA[0] * filterDimA[1] * filterDimA[2]));

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

            //Initialize layer bias
            cudnnFilterDescriptor_t linLayerBiasDesc;
            T_ELEM *linLayerBias;

            cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(cudnnHandle,
                                                        rnnDesc,
                                                        layer,
                                                        xDesc[0],
                                                        wDesc,
                                                        w,
                                                        linLayerID,
                                                        linLayerBiasDesc,
                                                        (void **)&linLayerBias));

            cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType, &format, &nbDims, filterDimA));

            initGPUData<T_ELEM>(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.0);

            cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
        }
    }

    // *********************************************************************************************************
    // Dynamic persistent RNN plan (if using this algo)
    // *********************************************************************************************************
    cudnnPersistentRNNPlan_t rnnPlan;
    if (algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        // Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
        //       minibatch or datatype don't change.
        cudnnErrCheck(cudnnCreatePersistentRNNPlan(rnnDesc, miniBatch, getDataType<T_ELEM>(), &rnnPlan));
        // Tell calls using this descriptor which plan to use.
        cudnnErrCheck(cudnnSetPersistentRNNPlan(rnnDesc, rnnPlan));
    }

    // *********************************************************************************************************
    // At this point all of the setup is done. We now need to pass through the RNN.
    // *********************************************************************************************************
    cudaErrCheck(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    float timeForward, timeBackward1, timeBackward2;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));

    cudaErrCheck(cudaEventRecord(start));

    // If we're not training we use this instead
    // cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
    // rnnDesc,
    // seqLength,
    // xDesc,
    // x,
    // hxDesc,
    // hx,
    // cxDesc,
    // cx,
    // wDesc,
    // w,
    // yDesc,
    // y,
    // hyDesc,
    // hy,
    // cyDesc,
    // cy,
    // workspace,
    // workSize));

    cudnnErrCheck(cudnnRNNForwardTraining(cudnnHandle,
                                          rnnDesc,
                                          seqLength,
                                          xDesc,
                                          x,
                                          hxDesc,
                                          hx,
                                          cxDesc,
                                          cx,
                                          wDesc,
                                          w,
                                          yDesc,
                                          y,
                                          hyDesc,
                                          hy,
                                          cyDesc,
                                          cy,
                                          workspace,
                                          workSize,
                                          reserveSpace,
                                          reserveSize));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeForward, start, stop));

    cudaErrCheck(cudaEventRecord(start));

    cudnnErrCheck(cudnnRNNBackwardData(cudnnHandle,
                                       rnnDesc,
                                       seqLength,
                                       yDesc,
                                       y,
                                       dyDesc,
                                       dy,
                                       dhyDesc,
                                       dhy,
                                       dcyDesc,
                                       dcy,
                                       wDesc,
                                       w,
                                       hxDesc,
                                       hx,
                                       cxDesc,
                                       cx,
                                       dxDesc,
                                       dx,
                                       dhxDesc,
                                       dhx,
                                       dcxDesc,
                                       dcx,
                                       workspace,
                                       workSize,
                                       reserveSpace,
                                       reserveSize));

    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackward1, start, stop));

    cudaErrCheck(cudaEventRecord(start));

    // cudnnRNNBackwardWeights adds to the data in dw.
    cudaErrCheck(cudaMemset(dw, 0, weightsSize));

    cudnnErrCheck(cudnnRNNBackwardWeights(cudnnHandle,
                                          rnnDesc,
                                          seqLength,
                                          xDesc,
                                          x,
                                          hxDesc,
                                          hx,
                                          yDesc,
                                          y,
                                          workspace,
                                          workSize,
                                          dwDesc,
                                          dw,
                                          reserveSpace,
                                          reserveSize));

    cudaErrCheck(cudaEventRecord(stop));

    cudaErrCheck(cudaEventSynchronize(stop));
    cudaErrCheck(cudaEventElapsedTime(&timeBackward2, start, stop));

    int numMats = 0;

    if (mode == CUDNN_RNN_RELU || mode == CUDNN_RNN_TANH) {
        numMats = 2;
    } else if (mode == CUDNN_LSTM) {
        numMats = 8;
    } else if (mode == CUDNN_GRU) {
        numMats = 6;
    }

    // Calculate FLOPS
    printf("Forward: %3.0f GFLOPS\n", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
    printf("Backward: %3.0f GFLOPS, ", numMats * 4ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
    printf("(%3.0f GFLOPS), ", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
    printf("(%3.0f GFLOPS)\n", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));

    // Save FLOPS to file
    fprintf(fp, "Forward: %3.0f GFLOPS\n", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeForward));
    fprintf(fp, "Backward: %3.0f GFLOPS, ", numMats * 4ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * (timeBackward1 + timeBackward2)));
    fprintf(fp, "(%3.0f GFLOPS), ", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward1));
    fprintf(fp, "(%3.0f GFLOPS)\n", numMats * 2ull * (bidirectional ? 2 : 1) * hiddenSize * hiddenSize * seqLength * miniBatch * numLayers / (1e6 * timeBackward2));

    // Make double-sure everything is finished before we copy for result checking.
    cudaDeviceSynchronize();

    // *********************************************************************************************************
    // Print checksums.
    // *********************************************************************************************************
    int biDirScale = (bidirectional ? 2 : 1);

    if (true) {
        T_ELEM *testOutputy;
        T_ELEM *testOutputhy;
        T_ELEM *testOutputcy;

        testOutputy = (T_ELEM *)malloc(seqLength * outputTensorSize * sizeof(T_ELEM));
        testOutputhy = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));
        testOutputcy = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));

        cudaErrCheck(cudaMemcpy(testOutputy, y, seqLength * outputTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        if (hy != NULL) {
            cudaErrCheck(cudaMemcpy(testOutputhy, hy, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }
        if (cy != NULL && mode == CUDNN_LSTM) {
            cudaErrCheck(cudaMemcpy(testOutputcy, cy, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }

        double checksumy = 0.f;
        double checksumhy = 0.f;
        double checksumcy = 0.f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumi = 0;
            double localSumh = 0;
            double localSumc = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < hiddenSize * biDirScale; i++) {
                    localSumi += (double) testOutputy[j * miniBatch * hiddenSize * biDirScale + m * hiddenSize * biDirScale + i];
                }
            }
            for (int j = 0; j < numLayers * biDirScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    if (hy != NULL) {
                        localSumh += (double) testOutputhy[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                    if ((cy != NULL) && (mode == CUDNN_LSTM)) {
                        localSumc += (double) testOutputcy[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
            }

            checksumy += localSumi;
            checksumhy += localSumh;
            checksumcy += localSumc;
        }

        printf("y checksum %E     ", checksumy);
        fprintf(fp, "y checksum %E     ", checksumy);
        if (mode == CUDNN_LSTM) {
            printf("cy checksum %E     ", checksumcy);
            fprintf(fp, "cy checksum %E     ", checksumcy);
        }
        printf("hy checksum %E\n", checksumhy);
        fprintf(fp, "hy checksum %E\n", checksumhy);

        free(testOutputy);
        free(testOutputcy);
        free(testOutputhy);
    }

    if (true) {
        T_ELEM *testOutputdx;
        T_ELEM *testOutputdhx;
        T_ELEM *testOutputdcx;

        testOutputdx = (T_ELEM *)malloc(seqLength * inputTensorSize * sizeof(T_ELEM));
        testOutputdhx = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));
        testOutputdcx = (T_ELEM *)malloc(hiddenTensorSize * sizeof(T_ELEM));

        cudaErrCheck(cudaMemcpy(testOutputdx, dx, seqLength * inputTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        if (dhx != NULL) {
            cudaErrCheck(cudaMemcpy(testOutputdhx, dhx, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }
        if ((dcx != NULL) && (mode == CUDNN_LSTM)) {
            cudaErrCheck(cudaMemcpy(testOutputdcx, dcx, hiddenTensorSize * sizeof(T_ELEM), cudaMemcpyDeviceToHost));
        }

        double checksumdx = 0.f;
        double checksumdhx = 0.f;
        double checksumdcx = 0.f;

        for (int m = 0; m < miniBatch; m++) {
            double localSumdx = 0;
            double localSumdhx = 0;
            double localSumdcx = 0;

            for (int j = 0; j < seqLength; j++) {
                for (int i = 0; i < inputSize; i++) {
                    localSumdx += (double) testOutputdx[j * miniBatch * inputSize + m * inputSize + i];
                }
            }

            for (int j = 0; j < numLayers * biDirScale; j++) {
                for (int i = 0; i < hiddenSize; i++) {
                    localSumdhx += (double) testOutputdhx[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    if (mode == CUDNN_LSTM) {
                        localSumdcx += (double) testOutputdcx[j * hiddenSize * miniBatch + m * hiddenSize + i];
                    }
                }
            }

            checksumdx += localSumdx;
            checksumdhx += localSumdhx;
            checksumdcx += localSumdcx;
        }

        printf("dx checksum %E    ", checksumdx);
        fprintf(fp, "dx checksum %E    ", checksumdx);
        if (mode == CUDNN_LSTM) {
            printf("dcx checksum %E    ", checksumdcx);
            fprintf(fp, "dcx checksum %E    ", checksumdcx);
        }
        printf("dhx checksum %E\n", checksumdhx);
        fprintf(fp, "dhx checksum %E\n", checksumdhx);

        free(testOutputdx);
        free(testOutputdhx);
        free(testOutputdcx);
    }

    if (true) {
        T_ELEM *testOutputdw;
        testOutputdw = (T_ELEM *)malloc(weightsSize);

        cudaErrCheck(cudaMemcpy(testOutputdw, dw, weightsSize, cudaMemcpyDeviceToHost));

        double checksumdw = 0.;

        for (int i = 0; i < weightsSize / sizeof(T_ELEM); i++) {
            checksumdw += (double) testOutputdw[i];
        }

        printf("dw checksum %E\n", checksumdw);
        fprintf(fp, "dw checksum %E\n", checksumdw);

        free(testOutputdw);
    }

    //Free all previously allocated memory, destroy all created cudnn descriptors
    if (algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
        cudnnDestroyPersistentRNNPlan(rnnPlan);
    }

    cudaFree(x);
    cudaFree(hx);
    cudaFree(cx);
    cudaFree(y);
    cudaFree(hy);
    cudaFree(cy);
    cudaFree(dx);
    cudaFree(dhx);
    cudaFree(dcx);
    cudaFree(dy);
    cudaFree(dhy);
    cudaFree(dcy);
    cudaFree(workspace);
    cudaFree(reserveSpace);
    cudaFree(w);
    cudaFree(dw);
    cudaFree(states);

    for (int i = 0; i < seqLength; i++) {
        cudnnDestroyTensorDescriptor(xDesc[i]);
        cudnnDestroyTensorDescriptor(yDesc[i]);
        cudnnDestroyTensorDescriptor(dxDesc[i]);
        cudnnDestroyTensorDescriptor(dyDesc[i]);
    }

    free(xDesc);
    free(yDesc);
    free(dxDesc);
    free(dyDesc);

    cudnnDestroyTensorDescriptor(hxDesc);
    cudnnDestroyTensorDescriptor(cxDesc);
    cudnnDestroyTensorDescriptor(hyDesc);
    cudnnDestroyTensorDescriptor(cyDesc);
    cudnnDestroyTensorDescriptor(dhxDesc);
    cudnnDestroyTensorDescriptor(dcxDesc);
    cudnnDestroyTensorDescriptor(dhyDesc);
    cudnnDestroyTensorDescriptor(dcyDesc);

    cudnnDestroyDropoutDescriptor(dropoutDesc);
    cudnnDestroyRNNDescriptor(rnnDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyFilterDescriptor(dwDesc);

    cudnnDestroy(cudnnHandle);
    printf("Output saved to result.txt\n");
    fclose(fp);
}

// Reads command line arguments and stores them in the proper variables
int parse_args(int argc,
               char *argv[],
               int &seqLength,
               int &numLayers,
               int &hiddenSize,
               int &inputSize,
               int &miniBatch,
               float &dropout,
               bool &bidirectional,
               cudnnRNNMode_t &mode,
               cudnnRNNAlgo_t &algo,
               cudnnDataType_t &dataType) {
    argc -= 1;
    argv++;
    while (argc) {
        if (argv[0][0] == '-') {
            if (strncmp(argv[0] + 1, "seqLength", strlen("seqLength")) == 0) {
                seqLength = atoi(argv[0] + 1 + strlen("seqLength"));
            } else if (strncmp(argv[0] + 1, "numLayers", strlen("numLayers")) == 0) {
                numLayers = atoi(argv[0] + 1 + strlen("numLayers"));
            } else if (strncmp(argv[0] + 1, "hiddenSize", strlen("hiddenSize")) == 0) {
                hiddenSize = atoi(argv[0] + 1 + strlen("hiddenSize"));
                inputSize = hiddenSize; // For now we fix inputSize = hiddenSize
            } else if (strncmp(argv[0] + 1, "miniBatch", strlen("miniBatch")) == 0) {
                miniBatch = atoi(argv[0] + 1 + strlen("miniBatch"));
            } else if (strncmp(argv[0] + 1, "dropout", strlen("dropout")) == 0) {
                char *p = argv[0] + 1 + strlen("dropout");
                sscanf(p, "%f", &dropout);
            } else if (strncmp(argv[0] + 1, "bidirectional", strlen("bidirectional")) == 0) {
                bidirectional = true;
            } else if (strncmp(argv[0] + 1, "mode", strlen("mode")) == 0) {
                mode = (cudnnRNNMode_t) atoi(argv[0] + 1 + strlen("mode"));
            } else if (strncmp(argv[0] + 1, "algo", strlen("algo")) == 0) {
                algo = (cudnnRNNAlgo_t) atoi(argv[0] + 1 + strlen("algo"));
            } else if (strncmp(argv[0] + 1, "Ps", strlen("Ps")) == 0) {
                dataType = CUDNN_DATA_FLOAT;
            } else if (strncmp(argv[0] + 1, "Pd", strlen("Pd")) == 0) {
                dataType = CUDNN_DATA_DOUBLE;
            } else if (strncmp(argv[0] + 1, "Ph", strlen("Ph")) == 0) {
                dataType = CUDNN_DATA_HALF;
            } else if (argv[0][1] == 'H') {
                printf("Usage\n");
                printf("  > ./RNN <flags>\n");
                printf("Command line flags\n");
                printf("  -seqLength<int>    : Specify sequence length\n");
                printf("  -numLayers<int>    : Specify number of layers\n");
                printf("  -hiddenSize<int>   : Specify hidden size\n");
                printf("  -miniBatch<int>    : Specify minibatch size\n");
                printf("  -dropout<float>    : Specify dropout probability\n");
                printf("  -bidirectional     : Switch to bidirectional instead of unidirectional RNN\n");
                printf("  -mode{0,1,2,3}     : Specify mode (ReLU, tanh, LSTM, GRU)\n");
                printf("  -persistent{0,1,2} : Specify recurrence algorithm (standard, persist dynamic, persist static)\n");
                printf("  -P{s,d,h}          : Specify data type precision (float, double, half)\n");
                printf("  -H                 : Display this help message\n");
                return 1;
            } else {
                printf("Improper command line flag! See \"./RNN -H\" for proper usage\n");
                return 1;
            }
        } else {
            printf("Improper command line flag! See \"./RNN -H\" for proper usage\n");
            return 1;
        }
        argc -= 1;
        argv++;
    }

    return 0;
}

// Checks whether the given parameters are supported by cuDNN, and prints them to the command line
int print_args(int seqLength,
               int numLayers,
               int hiddenSize,
               int inputSize,
               int miniBatch,
               float dropout,
               bool bidirectional,
               cudnnRNNMode_t mode,
               cudnnRNNAlgo_t algo,
               cudnnDataType_t dataType) {
    printf("seqLength  = %d\nnumLayers  = %d\nhiddenSize = %d\n", seqLength, numLayers, hiddenSize);
    printf("inputSize  = %d\nminiBatch  = %d\ndropout    = %.6f\n", inputSize, miniBatch, dropout);

    if (bidirectional) {
        printf("direction  = CUDNN_BIDIRECTIONAL\n");
    } else {
        printf("direction  = CUDNN_UNIDIRECTIONAL\n");
    }

    if (mode == CUDNN_RNN_RELU) {
        printf("mode       = CUDNN_RNN_RELU\n");
    } else if (mode == CUDNN_RNN_TANH) {
        printf("mode       = CUDNN_RNN_TANH\n");
    } else if (mode == CUDNN_LSTM) {
        printf("mode       = CUDNN_LSTM\n");
    } else if (mode == CUDNN_GRU) {
        printf("mode       = CUDNN_GRU\n");
    }

    if (algo == CUDNN_RNN_ALGO_STANDARD) {
        printf("algo       = CUDNN_RNN_ALGO_STANDARD\n");
    } else {
        // Persistent RNNs are only supported on Pascal+ GPUs.
        int device;
        struct cudaDeviceProp devProp;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&devProp, device);
        if (devProp.major < 6) {
            printf("!!! ERROR: Persistent RNNs are only supported on Pascal+ GPUs\n");
            return 1;
        }
        if (algo == CUDNN_RNN_ALGO_PERSIST_STATIC) {
            printf("algo       = CUDNN_RNN_ALGO_PERSIST_STATIC\n");
        } else if (algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC) {
            printf("algo       = CUDNN_RNN_ALGO_PERSIST_DYNAMIC\n");
        }
    }

    if (dataType == CUDNN_DATA_FLOAT) {
        printf("precision  = CUDNN_DATA_FLOAT\n");
    } else if (dataType == CUDNN_DATA_DOUBLE) {
        printf("precision  = CUDNN_DATA_DOUBLE\n");
        if (algo == CUDNN_RNN_ALGO_PERSIST_STATIC) {
            printf("!!! ERROR: Double precision is disabled for PERSIST_STATIC algorithm\n");
            return 1;
        }
    } else if (dataType == CUDNN_DATA_HALF) {
        printf("precision  = CUDNN_DATA_HALF\n");
    }

    printf("\n");
    return 0;
}

int main(int argc, char *argv[]) {
    // Default case is same as golden_1.txt
    int seqLength = 20;
    int numLayers = 2;
    int hiddenSize = 512;
    int inputSize = hiddenSize; // For now we fix inputSize = hiddenSize
    int miniBatch = 64;
    float dropout = 0;
    bool bidirectional = false;
    cudnnRNNMode_t mode = CUDNN_RNN_RELU;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;

    // Read in command line flags
    bool error = parse_args(argc,
                            argv,
                            seqLength,
                            numLayers,
                            hiddenSize,
                            inputSize,
                            miniBatch,
                            dropout,
                            bidirectional,
                            mode,
                            algo,
                            dataType);

    if (error) {
        return 0;
    }

    // Check and print arguments before performing test
    error = print_args(seqLength,
                       numLayers,
                       hiddenSize,
                       inputSize,
                       miniBatch,
                       dropout,
                       bidirectional,
                       mode,
                       algo,
                       dataType);

    if (error) {
        return 0;
    }

    // Perform test depending on precision
    if (dataType == CUDNN_DATA_FLOAT) {
        doTest<float>(seqLength, numLayers, hiddenSize, inputSize, miniBatch, dropout, bidirectional, mode, algo);
    } else if (dataType == CUDNN_DATA_DOUBLE) {
        doTest<double>(seqLength, numLayers, hiddenSize, inputSize, miniBatch, dropout, bidirectional, mode, algo);
    } else if (dataType == CUDNN_DATA_HALF) {
        doTest<half1>(seqLength, numLayers, hiddenSize, inputSize, miniBatch, dropout, bidirectional, mode, algo);
    }

    return 0;
}
