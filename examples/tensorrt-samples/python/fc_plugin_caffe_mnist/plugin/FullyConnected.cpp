/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <array>

#include "NvInfer.h"
#include "FullyConnected.h"

#define CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

namespace
{
    constexpr const char* FC_PLUGIN_VERSION{"1"};
    constexpr const char* FC_PLUGIN_NAME{"FCPlugin"};

    // Helpers to move data to/from the GPU.
    nvinfer1::Weights copyToDevice(const void* hostData, int count)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
        CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, deviceData, count};
    }

    nvinfer1::Weights makeDeviceCopy(nvinfer1::Weights deviceWeights)
    {
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, deviceWeights.count * sizeof(float)));
        CHECK(cudaMemcpy(deviceData, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToDevice));
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, deviceData, deviceWeights.count};
    }

    int copyFromDevice(char* hostBuffer, nvinfer1::Weights deviceWeights)
    {
        *reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
        CHECK(cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
        return sizeof(int) + deviceWeights.count * sizeof(float);
    }
}

nvinfer1::PluginFieldCollection FCPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> FCPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FCPluginCreator);

// In this simple case we're going to infer the number of output channels from the bias weights.
// The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
// divined from the caffe innards
FCPlugin::FCPlugin(const nvinfer1::Weights* weights, int nbWeights)
{
	assert(nbWeights == 2);
	mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
	mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
}

// Copy from existing device weights
FCPlugin::FCPlugin(const nvinfer1::Weights& deviceKernel, const nvinfer1::Weights& deviceBias)
{
	mKernelWeights = makeDeviceCopy(deviceKernel);
	mBiasWeights = makeDeviceCopy(deviceBias);
}

// Create the plugin at runtime from a byte stream.
FCPlugin::FCPlugin(const void* data, size_t length)
{
	const char* d = reinterpret_cast<const char*>(data);
	const char* check = d;
	// Deserialize kernel.
	const int kernelCount = reinterpret_cast<const int*>(d)[0];
	mKernelWeights = copyToDevice(d + sizeof(int), kernelCount);
	d += sizeof(int) + mKernelWeights.count * sizeof(float);
	// Deserialize bias.
	const int biasCount = reinterpret_cast<const int*>(d)[0];
	mBiasWeights = copyToDevice(d + sizeof(int), biasCount);
	d += sizeof(int) + mBiasWeights.count * sizeof(float);
	// Check that the sizes are what we expected.
	assert(d == check + length);
}

// Free buffers.
FCPlugin::~FCPlugin()
{
	cudaFree(const_cast<void*>(mKernelWeights.values));
	mKernelWeights.values = nullptr;
	cudaFree(const_cast<void*>(mBiasWeights.values));
	mBiasWeights.values = nullptr;
}

int FCPlugin::getNbOutputs() const
{
    return 1;
}

nvinfer1::Dims FCPlugin::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
	assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
	return nvinfer1::DimsCHW{static_cast<int>(mBiasWeights.count), 1, 1};
}

int FCPlugin::initialize()
{
	CHECK(cudnnCreate(&mCudnn));
	CHECK(cublasCreate(&mCublas));
	// Create cudnn tensor descriptors for bias addition.
	CHECK(cudnnCreateTensorDescriptor(&mSrcDescriptor));
	CHECK(cudnnCreateTensorDescriptor(&mDstDescriptor));
	return 0;
}

void FCPlugin::terminate()
{
	CHECK(cudnnDestroyTensorDescriptor(mSrcDescriptor));
	CHECK(cudnnDestroyTensorDescriptor(mDstDescriptor));
	CHECK(cublasDestroy(mCublas));
	CHECK(cudnnDestroy(mCudnn));
}

// This plugin requires no workspace memory during build time.
size_t FCPlugin::getWorkspaceSize(int maxBatchSize) const
{
    return 0;
}

int FCPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	int nbOutputChannels = mBiasWeights.count;
	int nbInputChannels = mKernelWeights.count / nbOutputChannels;
	constexpr float kONE = 1.0f, kZERO = 0.0f;
	// Do matrix multiplication.
	cublasSetStream(mCublas, stream);
	cudnnSetStream(mCudnn, stream);
	CHECK(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, nbOutputChannels, batchSize, nbInputChannels, &kONE,
			reinterpret_cast<const float*>(mKernelWeights.values), nbInputChannels,
			reinterpret_cast<const float*>(inputs[0]), nbInputChannels, &kZERO,
			reinterpret_cast<float*>(outputs[0]), nbOutputChannels));
    // Add bias.
	CHECK(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, nbOutputChannels, 1, 1));
	CHECK(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, nbOutputChannels, 1, 1));
	CHECK(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
	return 0;
}

size_t FCPlugin::getSerializationSize() const
{
	return sizeof(int) * 2 + mKernelWeights.count * sizeof(float) + mBiasWeights.count * sizeof(float);
}

void FCPlugin::serialize(void* buffer) const
{
	char* d = reinterpret_cast<char*>(buffer);
	const char* check = d;
	d += copyFromDevice(d, mKernelWeights);
	d += copyFromDevice(d, mBiasWeights);
	assert(d == check + getSerializationSize());
}

// For this sample, we'll only support float32 with NCHW.
bool FCPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const
{
	return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
}

const char* FCPlugin::getPluginType() const
{
    return FC_PLUGIN_NAME;
}

const char* FCPlugin::getPluginVersion() const
{
    return FC_PLUGIN_VERSION;
}

void FCPlugin::destroy()
{
    delete this;
}

IPluginV2Ext* FCPlugin::clone() const
{
    IPluginV2Ext* plugin = new FCPlugin(mKernelWeights, mBiasWeights);
    plugin->setPluginNamespace(mPluginNamespace.c_str());
    return plugin;
}

void FCPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* FCPlugin::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}

DataType FCPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

bool FCPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

bool FCPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void FCPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{
	assert(nbInputs == 1 && inputDims[0].d[1] == 1 && inputDims[0].d[2] == 1);
	assert(nbOutputs == 1 && outputDims[0].d[1] == 1 && outputDims[0].d[2] == 1);
	assert(mKernelWeights.count == inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2] * mBiasWeights.count);
}

void FCPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

void FCPlugin::detachFromContext() {}


//
// Plugin Creator
//

FCPluginCreator::FCPluginCreator()
{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

void FCPluginCreator::setPluginNamespace(const char* libNamespace)
{
    mNamespace = libNamespace;
}

const char* FCPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

const char* FCPluginCreator::getPluginName() const
{
    return FC_PLUGIN_NAME;
}

const char* FCPluginCreator::getPluginVersion() const
{
    return FC_PLUGIN_VERSION;
}

const PluginFieldCollection* FCPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* FCPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    std::array<Weights, 2> weights{};

    for (int i = 0; i < fc->nbFields; ++i)
    {
        std::string fieldName(fc->fields[i].name);
        if (fieldName.compare("kernel") == 0)
        {
            weights[0].values = fc->fields[i].data;
            weights[0].count = fc->fields[i].length;
            weights[0].type = nvinfer1::DataType::kFLOAT;
        }
        if (fieldName.compare("bias") == 0)
        {
            weights[1].values = fc->fields[i].data;
            weights[1].count = fc->fields[i].length;
            weights[1].type = nvinfer1::DataType::kFLOAT;
        }
    }

    return new FCPlugin(static_cast<void*>(weights.data()), weights.size());
}

IPluginV2Ext* FCPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    auto* plugin = new FCPlugin(serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
