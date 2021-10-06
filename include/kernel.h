#ifndef KERNEL_H
#define KERNEL_H

#include "cublas_v2.h"
//#include "plugin.h"
#include <cassert>

#include <cstdio>

void gpuPreProcess(void *d_input, void *d_output, int in_width, int in_height, int out_width, int out_height, float *means);
void gpuPreProcessLite(void *d_input, void *d_output, int out_width, int out_height, float *means);
void gpuPostProcess();
void gpuBBoxTranform(void *d_anchor, void *d_regression, void *d_output, int data_size);
void gpuFindMaxConfidence(void *d_classification, void *d_output, const int class_num, const int data_size);

#endif
