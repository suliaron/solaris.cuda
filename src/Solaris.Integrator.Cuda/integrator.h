#pragma once
#include <string>

#include "config.h"
#include "ode.h"
#ifdef TIMER
#include "timer.h"
#endif

class integrator
{
protected:
	ode& f;
	ttt_t dt;
	ttt_t dt_try;
	ttt_t dt_did;

	int_t n_failed_step;
	int_t n_step;

#ifdef TIMER
	timer	tmr;
#endif

public:
	integrator(ode& f, ttt_t dt);
	~integrator();

	int_t get_n_failed_step();
	int_t get_n_step();

	virtual ttt_t step() = 0;
	virtual std::string get_name() = 0;
};