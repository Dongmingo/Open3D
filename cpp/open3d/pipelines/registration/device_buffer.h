// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#ifndef __DEVICE_BUFFER_H__
#define __DEVICE_BUFFER_H__

#ifdef BUILD_CUDA_MODULE
#include <cuda_runtime.h>

#include "open3d/core/CUDAUtils.h"
#endif

// #include "macro.h"

template <typename T>
struct DeviceBuffer {
    DeviceBuffer() : data(nullptr), size(0) {}
    DeviceBuffer(size_t size) : data(nullptr), size(0) { allocate(size); }
    ~DeviceBuffer() { destroy(); }

    void allocate(size_t _size) {
        if (data && size >= _size) return;

        destroy();
        OPEN3D_CUDA_CHECK(cudaMalloc(&data, sizeof(T) * _size));
        size = _size;
    }

    void destroy() {
        if (data) OPEN3D_CUDA_CHECK(cudaFree(data));
        data = nullptr;
        size = 0;
    }

    void upload(const T* h_data) {
        OPEN3D_CUDA_CHECK(cudaMemcpy(data, h_data, sizeof(T) * size,
                                     cudaMemcpyHostToDevice));
    }

    void download(T* h_data) {
        OPEN3D_CUDA_CHECK(cudaMemcpy(h_data, data, sizeof(T) * size,
                                     cudaMemcpyDeviceToHost));
    }

    void copyTo(DeviceBuffer& rhs) const {
        OPEN3D_CUDA_CHECK(cudaMemcpy(rhs.data, data, sizeof(T) * size,
                                     cudaMemcpyDeviceToDevice));
    }

    void copyTo(T* rhs) const {
        OPEN3D_CUDA_CHECK(cudaMemcpy(rhs, data, sizeof(T) * size,
                                     cudaMemcpyDeviceToDevice));
    }

    void fillZero() {
        OPEN3D_CUDA_CHECK(cudaMemset(data, 0, sizeof(T) * size));
    }

    bool empty() const { return !(data && size > 0); }

    void assign(size_t size, const T* h_data) {
        allocate(size);
        upload(h_data);
    }

    T* data;
    size_t size;
};

#endif  // !__DEVICE_BUFFER_H__