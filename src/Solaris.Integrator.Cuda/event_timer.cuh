#pragma once
// include system
#include <assert.h>
#include <cstdint>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class event_timer {
public:
	event_timer() :	
		mStarted(false),
		mStopped(false)
	{
		cudaEventCreate(&mStart);
		cudaEventCreate(&mStop);
	}

	~event_timer()
	{
		cudaEventDestroy(mStart);
		cudaEventDestroy(mStop);
	}

	void start(cudaStream_t s = 0)
	{ 
		cudaEventRecord(mStart, s);
		mStarted = true;
		mStopped = false;
	}

	void stop(cudaStream_t s = 0)
	{
		assert(mStarted);
		cudaEventRecord(mStop, s);
		mStarted = false;
		mStopped = true;
	}

	float elapsed()
	{
		assert(mStopped);
		if (!mStopped) {
			return 0;
		}
		cudaEventSynchronize(mStop);
		float elapsed = 0.f;
		cudaEventElapsedTime(&elapsed, mStart, mStop);
		return elapsed;
	}

private:
	bool mStarted, mStopped;
	cudaEvent_t mStart, mStop;
};
