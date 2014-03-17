#pragma once

#include "config.h"

typedef enum gas_decrease
		{ 
			GAS_DECREASE_CONSTANT,
			GAS_DECREASE_LINEAR,
			GAS_DECREASE_EXPONENTIAL
		} gas_decrease_t;

class gas_disk
{
public:
	gas_decrease_t gas_decrease;		// The decrease type for the gas density

	gas_disk();
	gas_disk(var2_t rho, var2_t sch, var2_t eta, var2_t tau, gas_decrease_t gas_decrease, ttt_t t0, ttt_t t1, ttt_t timeScale);

	var_t	reduction_factor(ttt_t time);

	ttt_t	t0, t1, timeScale;

	var2_t	rho;
	var2_t	sch;
	var2_t	eta;
	var2_t	tau;

	// Input/Output streams
	friend std::ostream& operator<<(std::ostream&, gas_disk);
};
