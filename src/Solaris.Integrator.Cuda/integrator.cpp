#include <algorithm>

#include "integrator.h"

#define PASSED_DT_SIZE	100

integrator::integrator(ode& f, ttt_t dt) : 
	f(f),	
	dt(dt),
	has_data(false),
	passed_indexer(0),
	passed_dt(PASSED_DT_SIZE),
	n_failed_step(0),
	n_passed_step(0),
	n_tried_step(0),
	dt_did(0.0),
	dt_try(0.0)
{
}

integrator::~integrator()
{
}

int_t integrator::get_n_failed_step()
{
	return n_failed_step;
}

int_t integrator::get_n_passed_step()
{
	return n_passed_step;
}

int_t integrator::get_n_tried_step()
{
	return n_tried_step;
}

void integrator::update_counters(int iter)
{
	n_tried_step  += iter;
	n_failed_step += (iter - 1);
	n_passed_step++;

	if (passed_indexer >= PASSED_DT_SIZE)
	{
		has_data = true;
		passed_indexer = 0;
	}
	passed_dt[passed_indexer++] = dt_did;
}

ttt_t integrator::get_avg_dt()
{
	ttt_t avg_dt = 0.0;
	for (int i = 0; i < passed_dt.size(); i++)
	{
		avg_dt += passed_dt[i];
	}
	return (ttt_t)avg_dt/passed_dt.size();
}

ttt_t integrator::get_max_dt()
{
	return (ttt_t)*max_element(passed_dt.begin(), passed_dt.end());
}

ttt_t integrator::get_min_dt()
{
	return (ttt_t)*min_element(passed_dt.begin(), passed_dt.end());
}

void integrator::calculate_dt_try()
{
	ttt_t dt_max = get_max_dt();
	ttt_t dt_min = get_min_dt();
	ttt_t dt_avg = get_avg_dt();
	if (dt_try > dt_max)
	{
		dt_try = 0.8 * dt_max;
	}
	else if (dt_try < dt_max)
	{
		dt_try = 1.2 * dt_min;
	}
}

#undef PASSED_DT_SIZE