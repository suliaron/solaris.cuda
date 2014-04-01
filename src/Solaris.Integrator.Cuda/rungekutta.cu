// include system
#include <cstdio>

#include "util.h"
#include "rungekutta.h"

/*template<> var_t rungekutta<4>::a[] = { 1.0/2.0, 0.0, 1.0/2.0, 0.0, 0.0, 1.0 };
template<> var_t rungekutta<4>::b[] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
template<> ttt_t rungekutta<4>::c[] = { 0.0, 1.0/2.0, 1.0/2.0, 1.0 };

template<> var_t rungekutta<2>::a[] = { 1.0/2.0 };
template<> var_t rungekutta<2>::b[] = { 0.0, 1.0 };
template<> ttt_t rungekutta<2>::c[] = { 0.0, 1.0/2.0 };*/

template <class RKOrder>
rungekutta<RKOrder>::rungekutta(ode& f, ttt_t dt, bool adaptive, var_t tolerance) :
	integrator(f, dt),
	adaptive(adaptive),
	tolerance(tolerance),
	d_ytemp(f.get_order(), d_var_t()),
	d_f(f.get_order())
{
	int forder = f.get_order();

	// Allocate storage for differentials on the device	
	for (int i = 0; i < forder; i++)
	{
		d_ytemp[i].resize(f.h_y[i].size());
		d_f[i].resize(RKOrder::order);
		for (int r = 0; r < RKOrder::order; r++)
		{
			d_f[i][r].resize(f.h_y[i].size());
		}
	}
}

template <class RKOrder>
rungekutta<RKOrder>::~rungekutta()
{
}

template <class RKOrder>
ttt_t rungekutta<RKOrder>::step()
{
	int forder = f.get_order();

	int rr = 0;
	ttt_t ttemp;

	for (int r = 0; r < RKOrder::order; r++) {
		ttemp = f.t + RKOrder::c[r] * dt;

		for (int i = 0; i < forder; i++) {
			copy_vec(d_ytemp[i], f.d_y[i]);
		}
		
		// Calculate temporary values of the dependent variables
		for (int s = 0; s < r; s++) {
			for (int i = 0; i < forder && RKOrder::a[rr] != 0.0; i ++) {
				sum_vec(d_ytemp[i], d_ytemp[i], d_f[i][s], (var_t)(RKOrder::a[rr] * dt));
			}
			rr++;
		}

		for (int i = 0; i < forder; i++) {
			f.calculate_dy(i, r, ttemp, f.d_p, d_ytemp, d_f[i][r]);
		}
	}

	// Advance dependent variables
	for (int i = 0; i < forder; i++) {
		copy_vec(f.d_yout[i], f.d_y[i]);
		for (int r = 0; r < RKOrder::order; r++) {
			if (0.0 == RKOrder::b[r]) {
				continue;
			}
			sum_vec(f.d_yout[i], f.d_yout[i], d_f[i][r], (var_t)(RKOrder::b[r] * dt));
		}
	}
	n_failed_step += 0;
	n_step++;

	// Advance time
	f.tout = f.t + dt;
	f.swap_in_out();

	return dt;
}

template <class RKOrder>
std::string rungekutta<RKOrder>::get_name()
{
	if (adaptive)
	{
		return "a_" + RKOrder::name;
	}
	else
	{
		return RKOrder::name;
	}

	// TODO: delete
	/*switch (RKOrder::order) 
	{
	case 2:
		return adaptive ? "a_RungeKutta2" : "RungeKutta2";
	case 3:
		return adaptive ? "a_RungeKutta3" : "RungeKutta3";
	case 4:
		return adaptive ? "a_RungeKutta4" : "RungeKutta4";
	case 5:
		return adaptive ? "a_RungeKutta5" : "RungeKutta5";
	case 6:
		return adaptive ? "a_RungeKutta6" : "RungeKutta6";
	case 7:
		return adaptive ? "a_RungeKutta7" : "RungeKutta7";
	case 8:
		return adaptive ? "a_RungeKutta8" : "RungeKutta8";
	default:
		return "unknown";
	}*/
}

//template class rungekutta<2>;
//template class rungekutta<4>;

template class rungekutta<rk45>;