// include system
#include <Windows.h>

// include project
#include "timer.h"

timer::timer() :
	_start(0),
	_stop(0)
{
	cudaEventCreate(&_cuda_start);
	cudaEventCreate(&_cuda_stop);
}

timer::~timer()
{
	cudaEventDestroy(_cuda_start);
	cudaEventDestroy(_cuda_stop);
}

int64_t timer::start()
{
	_start = GetTimeMicro64();
	return _start;
}

int64_t timer::stop()
{
	_stop = GetTimeMicro64();
	return _stop;
}

void timer::reset()
{
	_start = 0;
	_stop = 0;
}

int64_t timer::ellapsed_time()
{
	return (_stop - _start);
}

void timer::cuda_start()
{
	cudaEventRecord(_cuda_start, 0);
}

void timer::cuda_stop()
{
	cudaEventRecord(_cuda_stop, 0);
	cudaEventSynchronize(_cuda_stop);
}

float timer::cuda_ellapsed_time()
{
	float elapsed = 0.f;
	cudaEventElapsedTime(&elapsed, _cuda_start, _cuda_stop);
	return elapsed;
}


int64_t timer::GetTimeMicro64()
{
	/* Windows */
	FILETIME ft;
	LARGE_INTEGER li;

	/* Get the amount of 100 nano seconds intervals elapsed since January 1, 1601 (UTC) and copy it
	* to a LARGE_INTEGER structure. */
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;

	uint64_t ret = li.QuadPart;
	ret -= 116444736000000000LL; /* Convert from file time to UNIX epoch time. */
	ret /= 10; /* From 100 nano seconds (10^-7) to 1 microsecond (10^-6) intervals */

	return ret;
}
