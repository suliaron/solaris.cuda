#pragma once
#include <string>

#include "config.h"
#include "integrator.h"
#include "ode.h"

class rkf7 : public integrator
{
public:
	static var_t a[];
	static var_t b[];
	static var_t bh[];
	static ttt_t c[];

private:
	//! The order of the embedded RK formulae
	int		RKOrder;
	//! The maximum number of the force calculation
	int		r_max;
	//! True if the method estimates the error and accordingly adjusts the step-size
	bool	adaptive;
	//! The maximum of the allowed local truncation error
	var_t	tolerance;

	//! Holds the derivatives for the differential equations
	std::vector<std::vector <d_var_t> >	d_f;
	//! Holds the temporary solution approximation along the step
	std::vector<d_var_t>				d_ytemp;
	//! Holds the leading local truncation error for each variable
	std::vector<d_var_t>				d_err;
	//! Holds the values against which the error is scaled
	std::vector<d_var_t>				d_yscale;

	dim3	grid;
	dim3	block;

	void calculate_grid(int nData, int threads_per_block);
	void call_calc_ytemp_for_fr_kernel(int r);
	void call_calc_y_np1_kernel();
	void call_calc_yscale_kernel();
	void call_calc_error_kernel();
	void call_calc_scalederror_kernel();

public:
	rkf7(ode& f, ttt_t, bool adaptive, var_t tolerance);
	~rkf7();

	ttt_t step();
	std::string get_name();
};
