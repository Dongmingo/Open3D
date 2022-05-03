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

#include "cusparse_wrapper.h"

/* Description: Gather of non-zero elements from dense vector y into
   sparse vector x. */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const float *y,
                                           float *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    int size = nnz;
    cusparseCreateSpVec(&vecX, size, nnz, (void*)xInd, (void*)xVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, size, (void*)y, CUDA_R_32F);
    return cusparseGather(handle, vecY, vecX);
    // return cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase);
}

/*
 * Low level API for GPU Cholesky
 *
 */
cusparseStatus_t CUSPARSEAPI cusparseXgthr(cusparseHandle_t handle,
                                           int nnz,
                                           const double *y,
                                           double *xVal,
                                           const int *xInd,
                                           cusparseIndexBase_t idxBase) {
    cusparseSpVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    int size = nnz;
    cusparseCreateSpVec(&vecX, size, nnz, (void*)xInd, (void*)xVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, size, (void*)y, CUDA_R_64F);
    return cusparseGather(handle, vecY, vecX);
    // return cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase);
}

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
                             size_t *workspaceInBytes) {
    return cusolverSpScsrcholBufferInfo(handle, n, nnzA, descrA, csrValA,
                                        csrRowPtrA, csrColIndA, info,
                                        internalDataInBytes, workspaceInBytes);
}

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
                             size_t *workspaceInBytes) {
    return cusolverSpDcsrcholBufferInfo(handle, n, nnzA, descrA, csrValA,
                                        csrRowPtrA, csrColIndA, info,
                                        internalDataInBytes, workspaceInBytes);
}

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const float *csrValA,
                         const int *csrRowPtrA,
                         const int *csrColIndA,
                         csrcholInfo_t info,
                         void *pBuffer) {
    return cusolverSpScsrcholFactor(handle, n, nnzA, descrA, csrValA,
                                    csrRowPtrA, csrColIndA, info, pBuffer);
}

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholFactor(cusolverSpHandle_t handle,
                         int n,
                         int nnzA,
                         const cusparseMatDescr_t descrA,
                         const double *csrValA,
                         const int *csrRowPtrA,
                         const int *csrColIndA,
                         csrcholInfo_t info,
                         void *pBuffer) {
    return cusolverSpDcsrcholFactor(handle, n, nnzA, descrA, csrValA,
                                    csrRowPtrA, csrColIndA, info, pBuffer);
}

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholZeroPivot(cusolverSpHandle_t handle,
                            csrcholInfo_t info,
                            float tol,
                            int *position) {
    return cusolverSpScsrcholZeroPivot(handle, info, tol, position);
}

cusolverStatus_t CUSOLVERAPI
cusolverSpXcsrcholZeroPivot(cusolverSpHandle_t handle,
                            csrcholInfo_t info,
                            double tol,
                            int *position) {
    return cusolverSpDcsrcholZeroPivot(handle, info, tol, position);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                                     int n,
                                                     const float *b,
                                                     float *x,
                                                     csrcholInfo_t info,
                                                     void *pBuffer) {
    return cusolverSpScsrcholSolve(handle, n, b, x, info, pBuffer);
}

cusolverStatus_t CUSOLVERAPI cusolverSpXcsrcholSolve(cusolverSpHandle_t handle,
                                                     int n,
                                                     const double *b,
                                                     double *x,
                                                     csrcholInfo_t info,
                                                     void *pBuffer) {
    return cusolverSpDcsrcholSolve(handle, n, b, x, info, pBuffer);
}