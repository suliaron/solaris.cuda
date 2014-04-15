// include system
#include <sstream>      // std::ostringstream
#include <cstdio>

// include CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// include project
#include "integrator_exception.h"
#include "rkf7.h"
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

ttt_t rkf7::c[] =  { 0.0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0.0, 1.0 };

var_t rkf7::a[] =  {	         0.0, 
					    2.0/27.0,
					    1.0/36.0,    1.0/12.0,
					    1.0/24.0,         0.0,   1.0/8.0,
					    5.0/12.0,         0.0, -25.0/16.0,   25.0/16.0,
					    1.0/20.0,         0.0,        0.0,    1.0/4.0,      1.0/5.0,
					  -25.0/108.0,        0.0,        0.0,  125.0/108.0,  -65.0/27.0,    125.0/54.0,
					   31.0/300.0,        0.0,        0.0,          0.0,   61.0/225.0,    -2.0/9.0,    13.0/900.0,
					          2.0,        0.0,        0.0,  -53.0/6.0,    704.0/45.0,   -107.0/9.0,    67.0/90.0,    3.0,
					  -91.0/108.0,        0.0,        0.0,   23.0/108.0, -976.0/135.0,   311.0/54.0,  -19.0/60.0,   17.0/6.0,  -1.0/12.0,
					 2383.0/4100.0,       0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0,
					    3.0/205.0,        0.0,        0.0,          0.0,           0.0,   -6.0/41.0,   -3.0/205.0,  -3.0/41.0,  3.0/41.0,   6.0/41.0, 0.0,
					-1777.0/4100.0,       0.0,        0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 12.0/41.0, 0.0, 1.0 };

var_t rkf7::b[]  = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0 };
var_t rkf7::bh[] = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 41.0/840.0, 41.0/840.0 };

// ytemp = yn + dt*(a10*f0)
static __global__
void calc_ytemp_for_f1_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, var_t a10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a10*f0[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a20*f0 + a21*f1)
static __global__
void calc_ytemp_for_f2_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f1, var_t a20, var_t a21)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a20*f0[tid] + a21*f0[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a30*f0 + a32*f2)
static __global__
void calc_ytemp_for_f3_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, var_t a30, var_t a32)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a30*f0[tid] + a32*f2[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a40*f0 + a42*f2 + a43*f3)
static __global__
void calc_ytemp_for_f4_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f2, const var_t *f3, var_t a40, var_t a42, var_t a43)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a40*f0[tid] + a42*f2[tid] + a43*f3[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a50*f0 + a53*f3 + a54*f4)
static __global__
void calc_ytemp_for_f5_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, var_t a50, var_t a53, var_t a54)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a50*f0[tid] + a53*f3[tid] + a54*f4[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a60*f0 + a63*f3 + a64*f4 + a65*f5)
static __global__
void calc_ytemp_for_f6_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, var_t a60, var_t a63, var_t a64, var_t a65)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a60*f0[tid] + a63*f3[tid] + a64*f4[tid] + a65*f5[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a70*f0 + a74*f4 + a75*f5 + a76*f6)
static __global__
void calc_ytemp_for_f7_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f4, const var_t *f5, const var_t *f6, var_t a70, var_t a74, var_t a75, var_t a76)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a70*f0[tid] + a74*f4[tid] + a75*f5[tid] + a76*f6[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a80*f0 + a83*f3 + a84*f4 + a85*f5 + a86*f6 + a87*f7)
static __global__
void calc_ytemp_for_f8_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, var_t a80, var_t a83, var_t a84, var_t a85, var_t a86, var_t a87)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a80*f0[tid] + a83*f3[tid] + a84*f4[tid] + a85*f5[tid] + a86*f6[tid] + a87*f7[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a90*f0 + a93*f3 + a94*f4 + a95*f5 + a96*f6 + a97*f7 + a98*f8)
static __global__
void calc_ytemp_for_f9_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, var_t a90, var_t a93, var_t a94, var_t a95, var_t a96, var_t a97, var_t a98)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a90*f0[tid] + a93*f3[tid] + a94*f4[tid] + a95*f5[tid] + a96*f6[tid] + a97*f7[tid] + a98*f8[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a100*f0 + a103*f3 + a104*f4 + a105*f5 + a106*f6 + a107*f7 + a108*f8 + a109*f9)
static __global__
void calc_ytemp_for_f10_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a100, var_t a103, var_t a104, var_t a105, var_t a106, var_t a107, var_t a108, var_t a109)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a100*f0[tid] + a103*f3[tid] + a104*f4[tid] + a105*f5[tid] + a106*f6[tid] + a107*f7[tid] + a108*f8[tid] + a109*f9[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a110*f0 + a115*f5 + a116*f6 + a117*f7 + a118*f8 + a119*f9)
static __global__
void calc_ytemp_for_f11_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, var_t a110, var_t a115, var_t a116, var_t a117, var_t a118, var_t a119)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a110*f0[tid] + a115*f5[tid] + a116*f6[tid] + a117*f7[tid] + a118*f8[tid] + a119*f9[tid]);
		tid += stride;
	}
}

// ytemp = yn + dt*(a120*f0 + a123*f3 + a124*f4 + a125*f5 + a126*f6 + a127*f7 + a128*f8 + a129*f9 + a1211*f11)
static __global__
void calc_ytemp_for_f12_kernel(int_t n, var_t *ytemp, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f3, const var_t *f4, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f11, var_t a120, var_t a123, var_t a124, var_t a125, var_t a126, var_t a127, var_t a128, var_t a129, var_t a1211)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		ytemp[tid] = y_n[tid] + dt * (a120*f0[tid] + a123*f3[tid] + a124*f4[tid] + a125*f5[tid] + a126*f6[tid] + a127*f7[tid] + a128*f8[tid] + a129*f9[tid] + f11[tid]);
		tid += stride;
	}
}

// For the scaling used to monitor accuracy
#define TINY	1.0e-30
// yscale = |y_n| + |dt * f0| + TINY
static __global__
void calc_yscale_kernel(int_t n, var_t *yscale, ttt_t dt, const var_t *y_n, const var_t *f0)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		yscale[tid] = fabs(y_n[tid]) + fabs(dt * f0[tid]) + TINY;
		tid += stride;
	}
}
#undef TINY

// err = f0 + f10 - f11 - f12
static __global__
void calc_error_kernel(int_t n, var_t *err, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		err[tid] = (f0[tid] + f10[tid] - f11[tid] - f12[tid]);
		tid += stride;
	}
}

// err = f0 + f10 - f11 - f12
static __global__
void calc_scalederror_kernel(int_t n, var_t *err, ttt_t dt, const var_t* yscale, const var_t *f0, const var_t *f10, const var_t *f11, const var_t *f12)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	var_t s = 41.0/840.0 * fabs(dt);
	while (n > tid) {
		err[tid] = (s * fabs(f0[tid] + f10[tid] - f11[tid] - f12[tid])) / yscale[tid];
		tid += stride;
	}
}

// y_n+1 = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10)
static __global__
void calc_y_np1_kernel(int_t n, var_t *y_np1, ttt_t dt, const var_t *y_n, const var_t *f0, const var_t *f5, const var_t *f6, const var_t *f7, const var_t *f8, const var_t *f9, const var_t *f10, var_t b0, var_t b5, var_t b6, var_t b7, var_t b8, var_t b9, var_t b10)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		y_np1[tid] = y_n[tid] + dt * (b0*f0[tid] + b5*f5[tid] + b6*(f6[tid] + f7[tid]) + b8*(f8[tid] + f9[tid]) + b10*f10[tid]);
		tid += stride;
	}
}





rkf7::rkf7(ode& f, ttt_t dt, bool adaptive, var_t tolerance) :
		integrator(f, dt),
		adaptive(adaptive),
		tolerance(tolerance),
		d_f(f.get_order()),
		d_ytemp(f.get_order(), d_var_t()),
		d_err(f.get_order(), d_var_t()),
		d_yscale(f.get_order(), d_var_t())
{
	RKOrder = 7;
	r_max = adaptive ? RKOrder + 6 : RKOrder + 4;
	int	forder = f.get_order();

	for (int i = 0; i < forder; i++) {
		d_ytemp[i].resize(f.d_y[i].size());
		if (adaptive) {
			d_err[i].resize(f.d_y[i].size());
			d_yscale[i].resize(f.d_y[i].size());
		}
		d_f[i].resize(r_max);
		for (int r = 0; r < r_max; r++) {
			d_f[i][r].resize(f.d_y[i].size());
		}
	}
}

void rkf7::call_calc_ytemp_for_fr_kernel(int r)
{
#ifdef TIMER
	cout << "call_calc_ytemp_for_fr_kernel start at " << tmr.start() << endl;
	tmr.cuda_start();
#endif
	int idx = 0;

	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		calculate_grid(n, THREADS_PER_BLOCK);

		var_t* y_n= f.d_y[i].data().get();
		var_t* ytemp= d_ytemp[i].data().get();
		var_t* f0 = d_f[i][0].data().get();
		var_t* f1 = d_f[i][1].data().get();
		var_t* f2 = d_f[i][2].data().get();
		var_t* f3 = d_f[i][3].data().get();
		var_t* f4 = d_f[i][4].data().get();
		var_t* f5 = d_f[i][5].data().get();
		var_t* f6 = d_f[i][6].data().get();
		var_t* f7 = d_f[i][7].data().get();
		var_t* f8 = d_f[i][8].data().get();
		var_t* f9 = d_f[i][9].data().get();
		var_t* f10= d_f[i][10].data().get();
		var_t* f11;
		if (adaptive) 
		{
			f11	= d_f[i][11].data().get();
		}

		switch (r) {
		case 1:
			idx = 1;		
			calc_ytemp_for_f1_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, a[idx]);
			break;
		case 2:
			idx = 2;
			calc_ytemp_for_f2_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f1, a[idx], a[idx+1]);
			break;
		case 3:
			idx = 4;
			calc_ytemp_for_f3_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f2, a[idx], a[idx+2]);
			break;
		case 4:
			idx = 7;
			calc_ytemp_for_f4_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f2, f3, a[idx], a[idx+2], a[idx+3]);
			break;
		case 5:
			idx = 11;
			calc_ytemp_for_f5_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, a[idx], a[idx+3], a[idx+4]);
			break;
		case 6:
			idx = 16;
			calc_ytemp_for_f6_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, f5, a[idx], a[idx+3], a[idx+4], a[idx+5]);
			break;
		case 7:
			idx = 22;
			calc_ytemp_for_f7_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f4, f5, f6, a[idx], a[idx+4], a[idx+5], a[idx+6]);
			break;
		case 8:
			idx = 29;
			calc_ytemp_for_f8_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7]);
			break;
		case 9:
			idx = 37;
			calc_ytemp_for_f9_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8]);
			break;
		case 10:
			idx = 46;
			calc_ytemp_for_f10_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
			break;
		case 11:
			idx = 56;
			calc_ytemp_for_f11_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f5, f6, f7, f8, f9, a[idx], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9]);
			break;
		case 12:
			idx = 67;
			calc_ytemp_for_f12_kernel<<<grid, block>>>(n, ytemp, dt_try, y_n, f0, f3, f4, f5, f6, f7, f8, f9, f11, a[idx], a[idx+3], a[idx+4], a[idx+5], a[idx+6], a[idx+7], a[idx+8], a[idx+9], a[idx+11]);
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
#ifdef TIMER
	tmr.cuda_stop();
	cout << "            ... stop at " << tmr.stop() << endl;
	cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
}

void rkf7::call_calc_yscale_kernel()
{
#ifdef TIMER
	cout << "call_calc_yscale_kernel start at " << tmr.start() << endl;
	tmr.cuda_start();
#endif
	for (int i = 0; i < f.get_order(); i++) {
		int n = f.d_y[i].size();
		var_t *yscale= d_yscale[i].data().get();
		var_t *y_n   = f.d_y[i].data().get();
		var_t *f0	 = d_f[i][0].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		calc_yscale_kernel<<<grid, block>>>(n, yscale, dt_try, y_n, f0);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_yscale_kernel failed");
		}
	}
#ifdef TIMER
	tmr.cuda_stop();
	cout << "            ... stop at " << tmr.stop() << endl;
	cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
}

void rkf7::call_calc_error_kernel()
{
#ifdef TIMER
	cout << "call_calc_error_kernel start at " << tmr.start() << endl;
	tmr.cuda_start();
#endif
	for (int i = 0; i < f.get_order(); i++) {
		int n = f.d_y[i].size();
		var_t *err = d_err[i].data().get();
		var_t *f0  = d_f[i][0].data().get();
		var_t *f10  = d_f[i][10].data().get();
		var_t *f11  = d_f[i][11].data().get();
		var_t *f12  = d_f[i][12].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		calc_error_kernel<<<grid, block>>>(n, err, f0, f10, f11, f12);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_error_kernel failed");
		}
	}
#ifdef TIMER
	tmr.cuda_stop();
	cout << "            ... stop at " << tmr.stop() << endl;
	cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
}

void rkf7::call_calc_scalederror_kernel()
{
#ifdef TIMER
	cout << "call_calc_scalederror_kernel start at " << tmr.start() << endl;
	tmr.cuda_start();
#endif
	for (int i = 0; i < f.get_order(); i++) {
		int n = f.d_y[i].size();
		var_t *err = d_err[i].data().get();
		var_t *yscale= d_yscale[i].data().get();
		var_t *f0  = d_f[i][0].data().get();
		var_t *f10  = d_f[i][10].data().get();
		var_t *f11  = d_f[i][11].data().get();
		var_t *f12  = d_f[i][12].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		calc_scalederror_kernel<<<grid, block>>>(n, err, dt_try, yscale, f0, f10, f11, f12);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_scalederror_kernel failed");
		}
	}
#ifdef TIMER
	tmr.cuda_stop();
	cout << "            ... stop at " << tmr.stop() << endl;
	cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
}

void rkf7::call_calc_y_np1_kernel()
{
#ifdef TIMER
	cout << "call_calc_y_np1_kernel start at " << tmr.start() << endl;
	tmr.cuda_start();
#endif
	for (int i = 0; i < f.get_order(); i++) {
		int n = f.d_y[i].size();
		var_t *y_n   = f.d_y[i].data().get();
		var_t *y_np1 = f.d_yout[i].data().get();
		var_t *f0	 = d_f[i][0].data().get();
		var_t *f5	 = d_f[i][5].data().get();
		var_t *f6	 = d_f[i][6].data().get();
		var_t *f7	 = d_f[i][7].data().get();
		var_t *f8	 = d_f[i][8].data().get();
		var_t *f9	 = d_f[i][9].data().get();
		var_t *f10   = d_f[i][10].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		calc_y_np1_kernel<<<grid, block>>>(n, y_np1, dt_try, y_n, f0, f5, f6, f7, f8, f9, f10, b[0], b[5], b[6], b[7], b[8], b[9], b[10]);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_y_np1_kernel failed");
		}
	}
#ifdef TIMER
	tmr.cuda_stop();
	cout << "            ... stop at " << tmr.stop() << endl;
	cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
}

void rkf7::calculate_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}

// constants for the Runge-Kutta-Fehlberg7(8) integrator
#define SAFETY	 0.9
#define PGROW	-0.2
#define PSHRNK	-0.25
#define ERRCON	 1.89e-4
ttt_t rkf7::step()
{
	int	forder = f.get_order();

	int r = 0;
	// Calculate f0 = f(tn, yn) = d_f[][0]
	ttt_t ttemp = f.t + c[r] * dt;
	for (int i = 0; i < forder; i++) {
		f.calculate_dy(i, r, ttemp, f.d_p, f.d_y, d_f[i][r]);
	}

	dt_try = dt;
	var_t max_err = 0.0;
	int_t iter = 0;
	do {
		dt_did = dt_try;
		// Calculate f1 = f(tn + c1 * dt, yn + a10 * dt * f0) = d_f[][1]
		// ...
		// Calculate f10 = f(tn + c10 * dt, yn + a10,0 * dt * f0 + ...) = d_f[][10]
		for (r = 1; r <= 10; r++) {
			ttemp = f.t + c[r] * dt_try;
			call_calc_ytemp_for_fr_kernel(r);
			for (int i = 0; i < forder; i++) {
				f.calculate_dy(i, r, ttemp, f.d_p, d_ytemp, d_f[i][r]);
			}
		}

		// y_(n+1) = yn + dt*(b0*f0 + b5*f5 + b6*f6 + b7*f7 + b8*f8 + b9*f9 + b10*f10) + O(dt^8)
		// f.d_yout = y_(n+1)
		call_calc_y_np1_kernel();
		if (adaptive) {
			call_calc_yscale_kernel();
			// Calculate f11 = f(tn + c11 * dt, yn + ...) = d_f[][11]
			// Calculate f12 = f(tn + c11 * dt, yn + ...) = d_f[][11]
			for (r = 11; r < r_max; r++) {
				ttemp = f.t + c[r] * dt_try;
				call_calc_ytemp_for_fr_kernel(r);
				for (int i = 0; i < forder; i++) {
					f.calculate_dy(i, r, ttemp, f.d_p, d_ytemp, d_f[i][r]);
				}
			}
			// calculate: d_err = (f0 + f10 - f11 - f12)
			//call_calc_error_kernel();
			//max_err = fabs(41.0/840 * dt_try * std::max(max_vec(d_err[0]), max_vec(d_err[1])));
			//dt_try *= 0.9 * pow(tolerance / max_err, 1.0/8.0);
			call_calc_scalederror_kernel();
			//max_err = fabs(std::max(max_vec(d_err[0]), max_vec(d_err[1]))) / tolerance;

#ifdef TIMER
			cout << "thrust::max_element start at " << tmr.start() << endl;
			tmr.cuda_start();
#endif
			var_t max_1 = *thrust::max_element(d_err[0].begin(), d_err[0].end()) / tolerance;
#ifdef TIMER
			cout << "            ... stop at " << tmr.stop() << endl;
			cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
#ifdef TIMER
			cout << "thrust::max_element start at " << tmr.start() << endl;
			tmr.cuda_start();
#endif
			var_t max_2 = *thrust::max_element(d_err[1].begin(), d_err[1].end()) / tolerance;
#ifdef TIMER
			cout << "            ... stop at " << tmr.stop() << endl;
			cout << "Took: " << tmr.ellapsed_time() << "\t" << tmr.cuda_ellapsed_time() << " [ms]" << endl;
#endif
			max_err = fabs(std::max(max_1, max_2));

			// The step failed, the required accuracy was not reached
			if (max_err > 1.0) {
				ttt_t dt_temp = SAFETY * dt_try * pow(max_err, PSHRNK);
				dt_try = (fabs(dt_temp) > fabs(0.1 * dt_try) ? dt_temp : 0.1 * dt_try);
			}
			// The step succeedded, the required accuracy was reached
			else {
				dt_try = (max_err > ERRCON ? (SAFETY * dt_try * pow(max_err, PGROW)) : (5.0 * dt_try));
			}
		}
		iter++;
	} while (adaptive && max_err > 1.0);
	n_failed_step += (iter - 1);
	n_step++;
	// Set the next step size
	dt = dt_try;

	f.tout = f.t + dt_did;
	f.swap_in_out();

	return dt_did;
}
#undef SAFETY
#undef PGROW
#undef PSHRNK
#undef ERRCON

string rkf7::get_name()
{
	return adaptive ? "a_RungeKuttaFehlberg78" : "RungeKuttaFehlberg78";
}
