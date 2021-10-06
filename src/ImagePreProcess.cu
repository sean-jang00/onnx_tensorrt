#include <cudnn.h>
#include "common.h"
template <typename Dtype, unsigned nthds_per_cta>
__global__ void PreProcess_kernel(
    const int nthreads,
    const unsigned char* data,
    float* new_data,
    float r_mean,float g_mean,float b_mean)
{
  for (int index = blockIdx.x * nthds_per_cta + threadIdx.x;
        index < nthreads;
        index += nthds_per_cta * gridDim.x)
  {
      int src_index = index*3;
      new_data[index] = ((float)data[src_index+2]/255.0f) - b_mean;//123.0f;// means[2];
      new_data[index+nthreads] = ((float)data[src_index+1]/255.f) - g_mean;//117.0f;//- means[1];
      new_data[index+(nthreads*2)] = ((float)data[src_index]/255.0f) - r_mean ;//104.0f;//- means[0];
      
  }
}
__global__ void resize_kernel(unsigned char *p_dst, unsigned char *p_src, int src_width, int src_height, int dst_width, int dst_height)
{
  int i = blockDim.y * blockIdx.y + threadIdx.y;
  int j = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < dst_height && j < dst_width)
  {
    int ii = i * src_height / dst_height;
    int jj = j * src_width / dst_width;
    int channel = 3;
    for(int c = 0; c < 3; c++)
      p_dst[(i*dst_width + j)*channel + c ] = p_src[ (ii*src_width + jj)*channel + c ];
  }
}
void gpuPreProcess(void *d_input, void *d_output, int in_width, int in_height, int out_width, int out_height, float *means)
{
  unsigned char* p_output_temp; 
  size_t buf_size = out_width*out_height*3*sizeof(unsigned char);
  cudaMalloc(&p_output_temp, buf_size);
  dim3 block(16, 16);
  dim3 grid((out_width+15)/16, (out_height+15)/16);
  resize_kernel<<< grid, block >>>(p_output_temp, (unsigned char*)d_input, in_width, in_height, out_width, out_height);
  int output_resolution = out_width * out_height;
  const int BS = 512;
  const int GS = (output_resolution + BS - 1) / BS;
  //PreProcess_kernel<float, BS><<<GS, BS, 0>>>(output_resolution,  (unsigned char*)d_input, (float*) d_output);
  PreProcess_kernel<float, BS><<<GS, BS, 0>>>(output_resolution,  p_output_temp, (float*) d_output, means[0], means[1], means[2]);

  cudaFree(p_output_temp);

  return;
}

void gpuPreProcessLite(void *d_input, void *d_output, int out_width, int out_height, float *means)
{
  int output_resolution = out_width * out_height;
  const int BS = 512;
  const int GS = (output_resolution + BS - 1) / BS;
  PreProcess_kernel<float, BS><<<GS, BS, 0>>>(output_resolution,  (unsigned char*)d_input, (float*) d_output, means[0], means[1], means[2]);

  return;
}
