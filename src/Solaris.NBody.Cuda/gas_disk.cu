#include <math.h>

#include "gas_disk.h"
#include "config.h"
#include "constants.h"

gas_disk::gas_disk() 
{
	gas_decrease = GAS_DENSITY_CONSTANT;
	t0 = 0.0;
	t1 = 0.0;
	e_folding_time = 0.0;

	alpha = 2.0e-3;
	eta.x = 2.0e-3;		eta.y =   1.0/2.0;
	rho.x = 1.0e-9;		rho.y = -11.0/4.0;		// g / cm^3
	sch.x = 5.0e-2;		sch.y =   5.0/4.0;
	tau.x = 2.0/3.0;	tau.y =   2.0;
	rho.x*= Constants::GramPerCm3ToSolarPerAu3; // M_sun / AU^3

	m_star = 1.0;
	mean_molecular_weight = 2.3;
	particle_diameter    = 3.0e-10 * Constants::MeterToAu;
	c_vth = sqrt((8.0 * Constants::Boltzman_CMU)/(Constants::Pi * mean_molecular_weight * Constants::ProtonMass_CMU));

	mfp.x = mean_molecular_weight * Constants::ProtonMass_CMU / (sqrt(2.0) * Constants::Pi * SQR(particle_diameter) * rho.x);
	mfp.y = -rho.y;

	temp.x = SQR(sch.x) * Constants::Gauss2 * m_star * mean_molecular_weight * Constants::ProtonMass_CMU / Constants::Boltzman_CMU;
	temp.y = 2.0 * sch.y - 3.0;
}

gas_disk::gas_disk(var2_t rho, var2_t sch, var2_t eta, var2_t tau, gas_decrease_t gas_decrease, ttt_t t0, ttt_t t1, ttt_t e_folding_time, var_t m_star) :
	rho(rho),
	sch(sch),
	eta(eta),
	tau(tau),
	gas_decrease(gas_decrease),
	t0(t0),
	t1(t1),
	e_folding_time(e_folding_time),
	m_star(m_star)
{
    mean_molecular_weight = 2.3;
	particle_diameter    = 3.0e-10 * Constants::MeterToAu;
	c_vth = sqrt((8.0 * Constants::Boltzman_CMU)/(Constants::Pi * mean_molecular_weight * Constants::ProtonMass_CMU));

	mfp.x = mean_molecular_weight * Constants::ProtonMass_CMU / (sqrt(2.0) * Constants::Pi * SQR(particle_diameter) * rho.x);
	mfp.y = -rho.y;

	temp.x = SQR(sch.x) * Constants::Gauss2 * m_star * mean_molecular_weight * Constants::ProtonMass_CMU / Constants::Boltzman_CMU;
	temp.y = 2.0 * sch.y - 3.0;
}

__host__ __device__
var_t	gas_disk::reduction_factor(ttt_t t)
{
	switch (gas_decrease) 
	{
	case GAS_DENSITY_CONSTANT:
		return 1.0;
	case GAS_DENSITY_DECREASE_LINEAR:
		if (t <= t0) {
			return 1.0;
		}
		else if (t0 < t && t <= t1 && t0 != t1) {
			return 1.0 - (t - t0)/(t1 - t0);
		}
		else {
			return 0.0;
		}
	case GAS_DENSITY_DECREASE_EXPONENTIAL:
		return exp(-(t - t0)/e_folding_time);
	default:
		return 1.0;
	}
}

std::ostream& operator<<(std::ostream& output, gas_disk gasDisk)
{
	output << "eta: " << gasDisk.eta.x << ", " << gasDisk.eta.y << std::endl;
	output << "rho: " << gasDisk.rho.x << ", " << gasDisk.rho.y << std::endl;
	output << "sch: " << gasDisk.sch.x << ", " << gasDisk.sch.y << std::endl;
	output << "tau: " << gasDisk.tau.x << ", " << gasDisk.tau.y << std::endl;
	output << "mfp: " << gasDisk.mfp.x << ", " << gasDisk.mfp.y << std::endl;
	output << "temp: " << gasDisk.temp.x << ", " << gasDisk.temp.y << std::endl;

	output << "gas_decrease: " << gasDisk.gas_decrease << std::endl;
	output << "          t0: " << gasDisk.t0 << " [d]" << std::endl;
	output << "          t1: " << gasDisk.t1 << " [d]" << std::endl;
	output << "   e_folding_time: " << gasDisk.e_folding_time << " [d]" << std::endl;

	output << "c_vth  : " << gasDisk.c_vth << std::endl;
	output << "alpha  : " << gasDisk.alpha << std::endl;
	output << "m_star : " << gasDisk.m_star << std::endl;
	output << "mean_molecular_weight : " << gasDisk.mean_molecular_weight << std::endl;
	output << "particle_diameter    : " << gasDisk.particle_diameter << std::endl;
		
	return output;
}
