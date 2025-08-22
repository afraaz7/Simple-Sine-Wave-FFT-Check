#include <iostream>
#include <complex>
#include <vector>
#include <math.h>
#include <cstdio>
#include <random>


constexpr double pi = 3.1415926535897932846;
/*

Simple FFT Check

Create a noisy signal and perform a Discrete Fourier Transform to extract the original frequencies.

Combine atmost two or three sine waves as a beginning.

*/

template<typename T>
void printVector(const std::vector <T>& vec) {
	for (const auto& element : vec) {
		std::cout << element << " ";
	}
	std::cout << std::endl;
}

inline std::vector<float> generateSignal(int freq1, int freq2, int freq3, float stepSize, int signalLength) {

	/*
	*  Constant Amplitude. 
	*  Generate Random noise to add to the signal.
	*/

	
	
	std::vector<float> result(signalLength);
	std::vector<float> time(signalLength);

	for (int i = 0; i < signalLength; i++) {
		time[i] = i * stepSize;
	}

	// Random Number Stuff

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<float> dist(0, 1);

	

	for (int i = 0; i < signalLength; i++) {
		result[i] = 0.8 + 0.7 * sin(2 * pi * freq1 * time[i]) + 0.7 * sin(2 * pi * freq2 * time[i]) + 
			0.7 * sin(2 * pi * freq3 * time[i]) + dist(gen);
	}

	return result;

}



