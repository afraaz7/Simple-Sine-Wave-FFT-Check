#include <cuda_runtime_api.h>
#include <cufft.h>
#include "Signal.h"
#include <complex>
// #include <matplot/matplot.h>



int main(int argc, char* argv[]) {
	cufftHandle planr2c;
	cudaStream_t stream = NULL;

	int batchSize = 4;
	int fftSize = 500;
	int signalLength = batchSize * fftSize;

	using scalar_type = float;
	using inputType = scalar_type;
	using outputType = std::complex<scalar_type>;

	std::vector<inputType> fftInput = generateSignal(50, 100, 200, 0.01, signalLength);
	std::vector<outputType> fftOutput((fftSize / 2 + 1) * batchSize);

	// Print out the first five elements of the input Array
	for (int i = 0; i < 5; i++) {
		std::printf("%f ", fftInput[i]);
	}

	inputType* d_input = nullptr;
	cufftComplex* d_output = nullptr;

	cufftCreate(&planr2c);
	cufftPlan1d(&planr2c, fftSize, CUFFT_R2C, batchSize);
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	cufftSetStream(planr2c, stream);


	// Create Device Arrays
	cudaMalloc(reinterpret_cast<void**> (&d_input), sizeof(inputType) * fftInput.size());

	cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(outputType) * fftOutput.size());

	cudaMemcpyAsync(d_input, fftInput.data(), sizeof(inputType) * fftInput.size(), cudaMemcpyHostToDevice, stream);

	// FOrward Transform

	cufftExecR2C(planr2c, d_input, d_output);

	//Copy data back to the host (CPU)
	cudaMemcpyAsync(fftOutput.data(), d_output, sizeof(outputType) * fftOutput.size(), cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	std::printf("Output Array after Forward FFT: \n");

	for (int i = 0; i < 5; i++) {
		std::printf("%f + %fj\n", fftOutput[i].real(), fftOutput[i].imag());
	}

	std::printf("=======================\n");

	cudaFree(d_input);
	cudaFree(d_output);
	
	cufftDestroy(planr2c);
	cudaStreamDestroy(stream);

	cudaDeviceReset();

	return EXIT_SUCCESS;

}