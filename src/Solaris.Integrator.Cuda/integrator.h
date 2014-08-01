#pragma once
#include <string>
#include <vector>

#include "config.h"
#include "ode.h"

using namespace std;

class integrator
{
protected:
	ode& f;
	ttt_t dt;
	ttt_t dt_try;
	ttt_t dt_did;

	int_t n_failed_step;
	int_t n_passed_step;
	int_t n_tried_step;

	bool has_data;
	int_t passed_indexer;
	vector<ttt_t>	passed_dt;

public:
	integrator(ode& f, ttt_t dt);
	~integrator();

	void update_counters(int iter);

	ttt_t get_avg_dt();
	ttt_t get_max_dt();
	ttt_t get_min_dt();
	void calculate_dt_try();

	int_t get_n_failed_step();
	int_t get_n_passed_step();
	int_t get_n_tried_step();

	virtual ttt_t step() = 0;
	virtual string get_name() = 0;
};