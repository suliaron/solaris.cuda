#pragma once

#include "config.h"

typedef enum gas_decrease
		{ 
			GAS_DENSITY_CONSTANT,
			GAS_DENSITY_DECREASE_LINEAR,
			GAS_DENSITY_DECREASE_EXPONENTIAL
		} gas_decrease_t;

class gas_disk
{
public:
	//! The decrease type for the gas density
	gas_decrease_t gas_decrease;

	//! Initilize a gas disk with default values
	gas_disk();
	gas_disk(var2_t rho, var2_t sch, var2_t eta, var2_t tau, gas_decrease_t gas_decrease, ttt_t t0, ttt_t t1, ttt_t e_folding_time, var_t m_star);

	__host__ __device__ 
	var_t	reduction_factor(ttt_t time);

	//! Time when the decrease of gas starts (for linear and exponential)
	ttt_t	t0; 
	//! Time when the linear decrease of the gas ends
	ttt_t	t1;
	//! The exponent for the exponential decrease
	ttt_t	e_folding_time;

	//! The density of the gas disk in the midplane (time dependent)
	var2_t	rho;
	//! The scale height of the gas disk
	var2_t	sch;
	//! Describes how the velocity of the gas differs from the circular velocity
	var2_t	eta;
	//! Describes the Type 2 migartion of the giant planets
	var2_t	tau;

	//! The mean free path of the gas molecules (calculated based on rho, time dependent)
	var2_t	mfp;
	//! The temperaterure of the gas (calculated based on sch)
	var2_t	temp;	

	//! Constant for computing the mean thermal velocity (calculated, constant)
	var_t	c_vth;

	//! The viscosity parameter for the Shakura & Sunyaev model (constant)
	var_t	alpha;
	//! The mean molecular weight in units of the proton mass (constant)
    var_t	mean_molecular_weight;
	//! The mean molecular diameter (constant)
	var_t	particle_diameter;
	//! The mass of the star (time dependent)
	var_t	m_star;

	// Input/Output streams
	friend std::ostream& operator<<(std::ostream&, gas_disk);
};
