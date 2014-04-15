#pragma once
// include system
#include <cstdint>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class timer
{
public:
	timer();
	~timer();

	int64_t start();
	int64_t stop();
	void reset();
	int64_t ellapsed_time();

	void cuda_start();
	void cuda_stop();
	void cuda_reset();
	float cuda_ellapsed_time();

private:
	int64_t GetTimeMicro64();

	int64_t _start;
	int64_t _stop;

	cudaEvent_t _cuda_start;
	cudaEvent_t _cuda_stop;

};
