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

#ifndef _FULLY_CONNECTED_H_
#define _FULLY_CONNECTED_H_

#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;

class FCPlugin: public IPluginV2Ext
{
public:
	FCPlugin(const Weights* weights, int nbWeights);

	FCPlugin(const Weights& deviceKernel, const Weights& deviceBias);

	FCPlugin(const void* data, size_t length);

	~FCPlugin();

	int getNbOutputs() const override;

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

	int initialize() override;

	void terminate() override;

	size_t getWorkspaceSize(int maxBatchSize) const override;

	int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	size_t getSerializationSize() const override;

	void serialize(void* buffer) const override;

	bool supportsFormat(DataType type, PluginFormat format) const override;

	const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    DataType getOutputDataType(int index, const DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
        const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

	void attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void detachFromContext() override;
private:
	std::string mPluginNamespace;

	cudnnHandle_t mCudnn;
	cublasHandle_t mCublas;
	Weights mKernelWeights{DataType::kFLOAT, nullptr}, mBiasWeights{DataType::kFLOAT, nullptr};
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};

class FCPluginCreator : public IPluginCreator
{
public:
	FCPluginCreator();

	~FCPluginCreator() override = default;

    void setPluginNamespace(const char* libNamespace) override;

    const char* getPluginNamespace() const override;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
private:
    std::string mNamespace;

    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

#endif //_FULLY_CONNECTED_H
