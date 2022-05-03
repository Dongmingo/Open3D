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

#ifndef __CUSPARSE_WRAPPER_H__
#define __CUSPARSE_WRAPPER_H__

#ifdef BUILD_CUDA_MODULE
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>
#endif

/* Description: Gather of non-zero elements from dense vector y into
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const float *y,
                                           float *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const double *y,
                                           double *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase);

/*
 * Low level API for GPU Cholesky
 *
 */
cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                             int n,
                             int nnzA,
                             const cusparseMatDescr_t descrA,
                             const float *csrValA,
                             const int *csrRowPtrA,
                             const int *csrColIndA,
                             csrcholInfo_t info,
                             size_t *internalDataInBytes,
                             size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholBufferInfo(cusolverSpHandle_t handle,
                             int n,
                             int nnzA,
                             const cusparseMatDescr_t descrA,
                             const double *csrValA,
                             const int *csrRowPtrA,
                             const int *csrColIndA,
                             csrcholInfo_t info,
                             size_t *internalDataInBytes,
                             size_t *workspaceInBytes);

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const float *csrValA,
                         const int *csrRowPtrA,
                         const int *csrColIndA,
                         csrcholInfo_t info,
                         void *pBuffer);

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const double *csrValA,
                         const int *csrRowPtrA,
                         const int *csrColIndA,
                         csrcholInfo_t info,
                         void *pBuffer);

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholZeroPivot(cusolverSpHandle_t handle,
                            csrcholInfo_t info,
                            float tol,
                            int *position);

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholZeroPivot(cusolverSpHandle_t handle,
                            csrcholInfo_t info,
                            double tol,
                            int *position);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                                     int n,
                                                     const float *b,
                                                     float *x,
                                                     csrcholInfo_t info,
                                                     void *pBuffer);

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                                     int n,
                                                     const double *b,
                                                     double *x,
                                                     csrcholInfo_t info,
                                                     void *pBuffer);

#endif  // !__CUSPARSE_WRAPPER_H__