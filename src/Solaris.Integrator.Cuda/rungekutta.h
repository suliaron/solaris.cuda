#pragma once
#include <string>

#include "config.h"
#include "integrator.h"
#include "ode.h"
#include "rkorder.h"

template <class RKOrder>
class rungekutta : public integrator
{
private:
	bool_t adaptive;
	var_t tolerance;
	
	std::vector< std::vector< d_var_t> > d_f;	// Differentials on the device
	std::vector<d_var_t> d_ytemp;				// Values on the device

public:
	rungekutta(ode& f, ttt_t dt, bool adaptive, var_t tolerance);
	~rungekutta();

	ttt_t step();
	virtual std::string get_name();
};
