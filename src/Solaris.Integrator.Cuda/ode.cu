#include <iomanip>
#include <iostream>

#include "ode.h"

using namespace std;

#ifdef STOP_WATCH

void ode::clear_elasped()
{
	for (int i = 0; i < ODE_KERNEL_N; i++)
	{
		elapsed[i] = 0.0;
	}
}

#endif

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
#ifdef STOP_WATCH
	s_watch.cuda_start();
#endif
	thrust::copy(d_p.begin(), d_p.end(), h_p.begin());
#ifdef STOP_WATCH
	s_watch.cuda_stop();
	elapsed[ODE_KERNEL_THRUST_COPY_TO_HOST] = s_watch.get_cuda_ellapsed_time();
#endif

	// Copy variables to the host
	for (unsigned int i = 0; i < h_y.size(); i++)
	{
#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		thrust::copy(d_y[i].begin(), d_y[i].end(), h_y[i].begin());
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[ODE_KERNEL_THRUST_COPY_TO_HOST] += s_watch.get_cuda_ellapsed_time();
#endif
	}
}

void ode::copy_to_device()
{
	// Copy parameters to the device
	d_p.resize(h_p.size());
#ifdef STOP_WATCH
	s_watch.cuda_start();
#endif
	thrust::copy(h_p.begin(), h_p.end(), d_p.begin());
#ifdef STOP_WATCH
	s_watch.cuda_stop();
	elapsed[ODE_KERNEL_THRUST_COPY_TO_DEVICE] = s_watch.get_cuda_ellapsed_time();
#endif

	// Copy variables to the device
	for (unsigned int i = 0; i < h_y.size(); i++)
	{
		d_y[i].resize(h_y[i].size());
#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		thrust::copy(h_y[i].begin(), h_y[i].end(), d_y[i].begin());
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[ODE_KERNEL_THRUST_COPY_TO_DEVICE] += s_watch.get_cuda_ellapsed_time();
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