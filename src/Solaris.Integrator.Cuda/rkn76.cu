// includes system 
#include <sstream>      // std::ostringstream
#include <cstdio>

// include CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// include project
#include "integrator_exception.h"
#include "rkn76.h"
#include "util.h"

#define THREADS_PER_BLOCK	256

static cudaError_t HandleError(cudaError_t cudaStatus, const char *file, int line)
{
    if (cudaSuccess != cudaStatus) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( cudaStatus ), file, line );
        return cudaStatus;
    }
	return cudaStatus;
}
#define HANDLE_ERROR(cudaStatus) (HandleError(cudaStatus, __FILE__, __LINE__))

#define	LAMBDA	1.0/20.0
#define sQ sqrt(21.0)
ttt_t rkn76::c[] = { 0.0, 1.0/10.0, 1.0/5.0, 3.0/8.0, 1.0/2.0, (7.0-sQ)/14.0, (7.0+sQ)/14.0, 1.0, 1.0 };
var_t rkn76::a[] = { 1.0/200.0,
						 1.0/150.0,                  1.0/75.0,
						 171.0/8192.0,              45.0/4096.0,                  315.0/8192.0,
						 5.0/288.0,                 25.0/528.0,                    25.0/672.0,                       16.0/693.0,
						 (1003.0-205.0*sQ)/12348.0,-25.0*(751.0-173.0*sQ)/90552.0, 25.0*(624.0-137.0*sQ)/43218.0,  -128.0*(361.0-79.0*sQ)/237699.0,      (3411.0-745.0*sQ)/24696.0,
						 (793.0+187.0*sQ)/12348.0, -25.0*(331.0+113.0*sQ)/90552.0, 25.0*(1044.0+247.0*sQ)/43218.0, -128.0*(14885.0+3779.0*sQ)/9745659.0, (3327.0+797.0*sQ)/24696.0,   -(581.0+127.0*sQ)/1722.0,
						-(157.0-3.0*sQ)/378.0,      25.0*(143.0-10.0*sQ)/2772.0,  -25.0*(876.0+55.0*sQ)/3969.0,    1280.0*(913.0+18.0*sQ)/596673.0,     -(1353.0+26.0*sQ)/2268.0,  7.0*(1777.0+377.0*sQ)/4428.0, 7.0*(5.0-sQ)/36.0,
						 1.0/20.0,                   0.0,                           0.0,                              0.0,                               8.0/45.0,                 7.0*(7.0+sQ)/360.0,           7.0*(7.0-sQ)/360.0, 0.0 };
var_t rkn76::bh[]= { 1.0/20.0, 0.0, 0.0, 0.0, 8.0/45.0, 7.0*(7.0+sQ)/360.0, 7.0*(7.0-sQ)/360.0,     0.0,    0.0 };
var_t rkn76::b[] = { 1.0/20.0, 0.0, 0.0, 0.0, 8.0/45.0, 7.0*(7.0+sQ)/360.0, 7.0*(7.0-sQ)/360.0, -LAMBDA, LAMBDA };
#undef sQ

// ytemp = y_n + dt*(a21*f1)
static __global__
void calc_ytemp_for_f2_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, var_t f1f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a31*f1 + a32*f2)
static __global__
void calc_ytemp_for_f3_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, var_t f1f, var_t f2f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a41*f1 + a42*f2 + a43*f3)
static __global__
void calc_ytemp_for_f4_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, var_t f1f, var_t f2f, var_t f3f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid] + f3f * f3[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a51*f1 + a52*f2 + a53*f3 + a54*f4)
static __global__
void calc_ytemp_for_f5_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, var_t f1f, var_t f2f, var_t f3f, var_t f4f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid] + f3f * f3[tid] + f4f * f4[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a61*f1 + a62*f2 + a63*f3 + a64*f4 + a65*f5)
static __global__
void calc_ytemp_for_f6_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, const var_t *f5, var_t f1f, var_t f2f, var_t f3f, var_t f4f, var_t f5f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid] + f3f * f3[tid] + f4f * f4[tid] + f5f * f5[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a71*f1 + a72*f2 + a73*f3 + a74*f4 + a75*f5 + a76*f6)
static __global__
void calc_ytemp_for_f7_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, var_t f1f, var_t f2f, var_t f3f, var_t f4f, var_t f5f, var_t f6f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid] + f3f * f3[tid] + f4f * f4[tid] + f5f * f5[tid] + f6f * f6[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a81*f1 + a82*f2 + a83*f3 + a84*f4 + a85*f5 + a86*f6 + a87*f7)
static __global__
void calc_ytemp_for_f8_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f2, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t f1f, var_t f2f, var_t f3f, var_t f4f, var_t f5f, var_t f6f, var_t f7f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f2f * f2[tid] + f3f * f3[tid] + f4f * f4[tid] + f5f * f5[tid] + f6f * f6[tid] + f7f * f7[tid];
		tid += stride;
	}
}

// ytemp = y_n + dt*(a91*f1 + a95*f5 + a96*f6 + a97*f7)
static __global__
void calc_ytemp_for_f9_kernel(int_t n, var_t *ytemp, const var_t *y_n, const var_t *f1, const var_t *f5, const var_t *f6, const var_t *f7, var_t f1f, var_t f5f, var_t f6f, var_t f7f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + f1f * f1[tid] + f5f * f5[tid] + f6f * f6[tid] + f7f * f7[tid];
		tid += stride;
	}
}

// y = y_n + dt*(bh1*f1 + bh5*f5 + bh6*f6 + bh7*f7)
static __global__
void calc_y_kernel(int_t n, var_t *y, const var_t *y_n, const var_t *f1, const var_t *f5, const var_t *f6, const var_t *f7, var_t f1f, var_t f5f, var_t f6f, var_t f7f)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		y[tid] = y_n[tid] + f1f * f1[tid] + f5f * f5[tid] + f6f * f6[tid] + f7f * f7[tid];
		tid += stride;
	}
}

static __global__
void calc_f8_sub_f9_kernel(int_t n, var_t* result, const var_t* f8, const var_t* f9)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		result[tid] = f8[tid] - f9[tid];
		tid += stride;
	}
}

void rkn76::call_calc_f8_sub_f9_kernel()
{
	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		var_t *err = d_err[i].data().get();
		var_t* f8	= d_f[i][7].data().get();
		var_t* f9	= d_f[i][8].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		calc_f8_sub_f9_kernel<<<grid, block>>>(n, err, f8, f9);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_f8_sub_f9_kernel failed");
		}
	}
}

void rkn76::call_calc_ytemp_for_fr_kernel(int r)
{
	int idx = 0;

	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		calculate_grid(n, THREADS_PER_BLOCK);

		var_t* y_n= f.d_y[i].data().get();
		var_t* f1 = d_f[i][0].data().get();
		var_t* f2 = d_f[i][1].data().get();
		var_t* f3 = d_f[i][2].data().get();
		var_t* f4 = d_f[i][3].data().get();
		var_t* f5 = d_f[i][4].data().get();
		var_t* f6 = d_f[i][5].data().get();
		var_t* f7 = d_f[i][6].data().get();
		var_t* f8;
		if (adaptive) {
			f8 = d_f[i][7].data().get();
		}
		switch (r) {
		case 1:
			idx = 0;		
			calc_ytemp_for_f2_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, a[idx]*dt_try);
			break;
		case 2:
			idx = 1;
			calc_ytemp_for_f3_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, a[idx]*dt_try, a[idx+1]*dt_try);
			break;
		case 3:
			idx = 3;
			calc_ytemp_for_f4_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, f3, a[idx]*dt_try, a[idx+1]*dt_try, a[idx+2]*dt_try);
			break;
		case 4:
			idx = 6;
			calc_ytemp_for_f5_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, f3, f4, a[idx]*dt_try, a[idx+1]*dt_try, a[idx+2]*dt_try, a[idx+3]*dt_try);
			break;
		case 5:
			idx = 10;
			calc_ytemp_for_f6_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, f3, f4, f5, a[idx]*dt_try, a[idx+1]*dt_try, a[idx+2]*dt_try, a[idx+3]*dt_try, a[idx+4]*dt_try);
			break;
		case 6:
			idx = 15;
			calc_ytemp_for_f7_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, f3, f4, f5, f6, a[idx]*dt_try, a[idx+1]*dt_try, a[idx+2]*dt_try, a[idx+3]*dt_try, a[idx+4]*dt_try, a[idx+5]*dt_try);
			break;
		case 7:
			idx = 21;
			calc_ytemp_for_f8_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f2, f3, f4, f5, f6, f7, a[idx]*dt_try, a[idx+1]*dt_try, a[idx+2]*dt_try, a[idx+3]*dt_try, a[idx+4]*dt_try, a[idx+5]*dt_try, a[idx+6]*dt_try);
			break;
		case 8:
			idx = 28;
			calc_ytemp_for_f9_kernel<<<grid, block>>>(n, d_ytemp[i].data().get(), y_n, f1, f5, f6, f7, a[idx]*dt_try, a[idx+4]*dt_try, a[idx+5]*dt_try, a[idx+6]*dt_try);
			break;
		default:
			ostringstream msg("call_calc_ytemp_for_fr_kernel() function was called with invalid parameter: ", ostringstream::ate);
			msg << r+1 << "!";
			throw integrator_exception(msg.str());
		}
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			ostringstream msg("calc_ytemp_for_f", ostringstream::ate);
			msg << r+1 << "_kernel failed";
			throw integrator_exception(msg.str());
		}
	}
}

void rkn76::call_calc_y_kernel()
{
	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		calculate_grid(n, THREADS_PER_BLOCK);

		var_t* y_n= f.d_y[i].data().get();
		var_t *y  = f.d_yout[i].data().get();
		var_t* f1 = d_f[i][0].data().get();
		var_t* f5 = d_f[i][4].data().get();
		var_t* f6 = d_f[i][5].data().get();
		var_t* f7 = d_f[i][6].data().get();
		calc_y_kernel<<<grid, block>>>(n, y, y_n, f1, f5, f6, f7, b[0]*dt_try, b[4]*dt_try, b[5]*dt_try, b[6]*dt_try);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_y_kernel failed");
		}
	}
}

rkn76::rkn76(ode& f, ttt_t dt, bool adaptive, var_t tolerance) :
		integrator(f, dt),
		adaptive(adaptive),
		tolerance(tolerance),
		d_f(f.get_order()),
		d_ytemp(f.get_order(), d_var_t()),
		d_err(f.get_order(), d_var_t())
{
	RKOrder = 7;
	r_max = adaptive ? RKOrder + 2 : RKOrder;
	int forder = f.get_order();

	for (int i = 0; i < forder; i++) {
		int size = f.d_y[i].size();
		d_ytemp[i].resize(size);
		if (adaptive) {
			d_err[i].resize(size);
		}
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) {
			d_f[i][r].resize(size);
		}
	}
}

void rkn76::calculate_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}

ttt_t rkn76::step()
{
	int	forder = f.get_order();

	int r = 0;
	// Calculate f1 = f(tn, yn) = d_f[][0]
	ttt_t ttemp = f.t + c[r] * dt;
	for (int i = 0; i < forder; i++) {
		f.calculate_dy(i, r, ttemp, f.d_p, f.d_y, d_f[i][r]);
	}

	dt_try = dt;
	var_t max_err = 0.0;
	int_t iter = 0;
	do {
		dt_did = dt_try;
		// Calculate f2 = f(tn + c2 * dt, yn + a21 * dt * f1) = d_f[][1]
		// Calculate f3 = f(tn + c3 * dt, yn + a31 * dt * f1 + ...) = d_f[][2]
		// Calculate f4 = f(tn + c4 * dt, yn + a41 * dt * f1 + ...) = d_f[][3]
		// ...
		// Calculate f7 = f(tn + c7 * dt, yn + a71 * dt * f1 + ...) = d_f[][6]
		for (r = 1; r < RKOrder; r++) {
			ttemp = f.t + c[r] * dt_try;
			call_calc_ytemp_for_fr_kernel(r);
			for (int i = 0; i < forder; i++) {
				f.calculate_dy(i, r, ttemp, f.d_p, d_ytemp, d_f[i][r]);
			}
		}

		if (adaptive) {
			// Calculate f8 = f(tn + c8 * dt, yn + a81 * dt * f1 + ...) = d_f[][7]
			// Calculate f9 = f(tn + c9 * dt, yn + a91 * dt * f1 + ...) = d_f[][8]
			for (r = RKOrder; r < r_max; r++) {
				ttemp = f.t + c[r] * dt_try;
				call_calc_ytemp_for_fr_kernel(r);
				for (int i = 0; i < forder; i++) {
					f.calculate_dy(i, r, ttemp, f.d_p, r == r_max - 1 ? f.d_yout : d_ytemp, d_f[i][r]);
				}
			}
			// calculate d_err = f8 - f9
			call_calc_f8_sub_f9_kernel();
			max_err = fabs(dt_try*LAMBDA*std::max(max_vec(d_err[0]), max_vec(d_err[1])));
			dt_try *= 0.9 * pow(tolerance / max_err, 1.0/8.0);
		}
		else {
			call_calc_y_kernel();
		}
		iter++;
	} while(adaptive && max_err > tolerance);
	n_failed_step += (iter - 1);
	n_step++;
	// Set the next step size
	dt = dt_try;

	f.tout = f.t + dt_did;
	f.swap_in_out();

	return dt_did;
}

string rkn76::get_name()
{
	return adaptive ? "a_optRungeKuttaNystrom76" : "optRungeKuttaNystrom76";
}

#undef LAMBDA
