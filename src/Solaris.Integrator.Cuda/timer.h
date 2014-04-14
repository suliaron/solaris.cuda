#pragma once
#include <cstdint>

class timer
{
public:
	timer();
	~timer();

	int64_t start();
	int64_t stop();
	void reset();
	int64_t ellapsed_time();

private:
	int64_t GetTimeMicro64();

	int64_t _start;
	int64_t _stop;
};
