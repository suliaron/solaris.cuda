#include <iomanip>
#include <iostream>

#include "ode.h"

using namespace std;

ode::ode(int order, ttt_t t) :
	t(t),
	h_y(order, h_var_t()),
	d_y(order, d_var_t()),
	d_yout(order, d_var_t())
{
}

ode::~ode()
{
}

int ode::get_order()
{
	return h_y.size();
}

void ode::copy_to_host()
{
	// Copy parameters to the host
#ifdef TIMER
	tmr.cuda_start();
#endif
	thrust::copy(d_p.begin(), d_p.end(), h_p.begin());
#ifdef TIMER
	tmr.cuda_stop();
	cout << setw(50) << "thrust::copy() took " << setw(20) << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif

	// Copy variables to the host
	for (unsigned int i = 0; i < h_y.size(); i++)
	{
#ifdef TIMER
		tmr.cuda_start();
#endif
		thrust::copy(d_y[i].begin(), d_y[i].end(), h_y[i].begin());
#ifdef TIMER
		tmr.cuda_stop();
		cout << setw(50) << "thrust::copy() took " << setw(20) << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
	}
}

void ode::copy_to_device()
{
	// Copy parameters to the device
	d_p.resize(h_p.size());
#ifdef TIMER
	tmr.cuda_start();
#endif
	thrust::copy(h_p.begin(), h_p.end(), d_p.begin());
#ifdef TIMER
	tmr.cuda_stop();
	cout << setw(50) << "thrust::copy() took " << setw(20) << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif

	// Copy variables to the device
	for (unsigned int i = 0; i < h_y.size(); i++)
	{
		d_y[i].resize(h_y[i].size());
#ifdef TIMER
		tmr.cuda_start();
#endif
		thrust::copy(h_y[i].begin(), h_y[i].end(), d_y[i].begin());
#ifdef TIMER
	tmr.cuda_stop();
	cout << setw(50) << "thrust::copy() took " << setw(20) << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif

		d_yout[i].resize(h_y[i].size());
	}
}

void ode::swap_in_out()
{
	// Swap values
	for (unsigned int i = 0; i < d_y.size(); i ++)
	{
		d_yout[i].swap(d_y[i]);
	}

	// Swap time
	std::swap(t, tout);
}