#pragma once
// include system
#include <cstdint>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes project
#include "config.h"

class stop_watch
{
public:
	stop_watch();
	~stop_watch();

	int64_t start();
	int64_t stop();
	int64_t get_ellapsed_time();

	void cuda_start();
	void cuda_stop();
	var_t get_cuda_ellapsed_time();

private:
	int64_t GetTimeMicro64();

	int64_t _start;
	int64_t _stop;

	cudaEvent_t _cuda_start;
	cudaEvent_t _cuda_stop;

};
