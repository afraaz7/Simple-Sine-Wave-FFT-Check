#include <iostream>
#include "cuda_runtime.h"

int main(){

        int devCount;
        cudaGetDeviceCount(&devCount);

        std::cout << "Number of Devices: " << devCount << std::endl;

        cudaDeviceProp devProp;

        cudGetDeviceProperties(&devProp, 0);
        float pkbdwidth = 2.0 * devProp.memoryClockRate * (devProp.memoryBusWidth/8) / 1e6 << " GB/s" << std::endl;
        std::cout << "Name of the Device: " << devProp.name << std::endl;
        std::cout << "Max Threads Per Block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Memory Bus Width: " << devProp.memoryBusWidth << std::endl;
        std::cout << "Peak Memory Bandwidth: " << pkbdwidth << std::endl;
        std::cout << "Memory Clock Rate: " << devProp.memoryClockRate << std::endl;
        std::cout << "Number of Streaming MultiProcessors " << devProp.multiProcessorCount << std::endl;
        std::cout << "Warp Size : " << devProp.warpSize << std::endl;
        std::cout << "Max Registers per Block: " << devProp.regsPerBlock << std::endl;
        std::cout << "Max threads along each dimension [x, y, z]: [" << devProp.maxThreadsDim[0] << ", " << devProp.maxThreadsDim[1] << ", " << devProp.maxThreadsDim[2] << "]" << std::endl;


        return 0;

}

