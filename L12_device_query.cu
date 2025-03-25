#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <string>

int main()
{
    int deviceCount = 0;

    // get GPU counter for current device
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0)
    {
        printf("there are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA capable device(s)\n", deviceCount);
    }

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaSetDevice(dev);
        //initilize current device property variable
        cudaDeviceProp deviceProp;
        cudaDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // GPU memory
        printf("total amount of global memory: %.0f Mbytes "
               "(%llu bytes)\n",
               static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
               (unsigned long long)deviceProp.totalGLobalMem);
        
        // clock fre
        printf("GPU max clock frequenz: %.0f Mhz (%0.2f" "Ghz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

        // L2 cache
        printf("L2 cache Size: %d bytes\n", deviceProp.l2CacheSize);

        // high fre used
        printf("total amount of shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("total amount of shared memory per multiprocessor: %zu bytes\n", deviceProp.sharedMemPerMultiprocessor);
        printf("total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("warp size: %d\n", deviceProp.warpSize);
        printf("maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("maximum dimension size of a block size (x, y, z): (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("maximum dimension size of a grid size (x, y, z): (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    }
    return 0;
}