// include system
#include <cstdio>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// include project
#include "integrator_exception.h"
#include "midpoint.h"

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

var_t midpoint::a[] = {1.0/2.0};
var_t midpoint::b[] = {0.0, 1.0};
ttt_t midpoint::c[] = {0.0, 1.0/2.0};

// result = a + b_factor * b
static __global__
void sum_vector_kernel(int_t n, var_t* result, const var_t* a, const var_t* b, var_t b_factor)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	while (n > tid) {
		result[tid] = a[tid] + b_factor * b[tid];
		tid += stride;
	}
}

void midpoint::calculate_grid(int nData, int threads_per_block)
{
	int	nThread = std::min(threads_per_block, nData);
	int	nBlock = (nData + nThread - 1)/nThread;
	grid.x  = nBlock;
	block.x = nThread;
}

void midpoint::calc_ytemp_for_k2()
{
	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		var_t *y_n	= f.d_y[i].data().get();
		var_t *ytemp= d_ytemp[i].data().get();
		var_t *k1	= d_f[i][0].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		sum_vector_kernel<<<grid, block>>>(n, ytemp, y_n, k1, a[0] * dt);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_ytemp_for_k2_kernel failed");
		}
	}
}

void midpoint::calc_y_np1()
{
	for (int i = 0; i < f.get_order(); i++) {
		int n		= f.d_y[i].size();
		var_t *y_n	= f.d_y[i].data().get();
		var_t *y_np1= f.d_yout[i].data().get();
		var_t *k2	= d_f[i][1].data().get();

		calculate_grid(n, THREADS_PER_BLOCK);
		sum_vector_kernel<<<grid, block>>>(n, y_np1, y_n, k2, b[1] * dt);
		cudaError cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw integrator_exception("calc_final_y_kernel failed");
		}
	}
}

midpoint::midpoint(ode& f, ttt_t dt, bool adaptive, var_t tolerance) :
		integrator(f, dt),
		adaptive(adaptive),
		tolerance(tolerance),
		d_f(f.get_order()),
		d_ytemp(f.get_order(), d_var_t())
{
	RKOrder = 2;
	int forder = f.get_order();

	for (int i = 0; i < forder; i++) {
		d_ytemp[i].resize(f.d_y[i].size());
		d_f[i].resize(RKOrder);
		for (int r = 0; r < RKOrder; r++) {
			d_f[i][r].resize(f.d_y[i].size());
		} 
	}
}

ttt_t	midpoint::step()
{
	int	forder = f.get_order();

	int r = 0;

	dt_did = dt;
	// Calculate k1 = f(tn, yn) = d_f[][0]
	ttt_t ttemp = f.t + c[r] * dt;
	for (int i = 0; i < forder; i++) {
		f.calculate_dy(i, r, ttemp, f.d_p, f.d_y, d_f[i][r]);
	}

	r = 1;
	// Calculate k2 = f(tn + c1 * dt, yn + a21 * dt * k1) = d_f[][1]
	calc_ytemp_for_k2();
	ttemp = f.t + c[r] * dt;
	for (int i = 0; i < forder; i++) {
		f.calculate_dy(i, r, ttemp, f.d_p, d_ytemp, d_f[i][r]); 
	}
	update_counters(1);

	calc_y_np1();

	f.tout = f.t + dt;
	f.swap_in_out();

	return dt;
}

string midpoint::get_name()
{
	return adaptive ? "a_Midpoint" : "Midpoint";
}
