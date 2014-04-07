// includes, system 
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "config.h"
#include "constants.h" 
#include "gas_disk.h"
#include "nbody.h"
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "ode.h"
#include "options.h"
#include "pp_disk.h"

using namespace std;

static cudaError_t HandleError(cudaError_t cudaStatus, const char *file, int line)
{
    if (cudaSuccess != cudaStatus) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( cudaStatus ), file, line );
        return cudaStatus;
    }
	return cudaStatus;
}
#define HANDLE_ERROR(cudaStatus) (HandleError(cudaStatus, __FILE__, __LINE__))

__host__ __device__
var_t	reduction_factor(gas_disk *gasDisk, ttt_t t)
{
	switch (gasDisk->gas_decrease) 
	{
	case GAS_DENSITY_CONSTANT:
		return 1.0;
	case GAS_DENSITY_DECREASE_LINEAR:
		if (t <= gasDisk->t0) {
			return 1.0;
		}
		else if (gasDisk->t0 < t && t <= gasDisk->t1 && gasDisk->t0 != gasDisk->t1) {
			return 1.0 - (t - gasDisk->t0)/(gasDisk->t1 - gasDisk->t0);
		}
		else {
			return 0.0;
		}
	case GAS_DENSITY_DECREASE_EXPONENTIAL:
		if (t <= gasDisk->t0) {
			return 1.0;
		}
		else {
			return exp(-(t - gasDisk->t0)/gasDisk->e_folding_time);
		}
	default:
		return 1.0;
	}
}


__global__
void print_gas_disc(gas_disk *gasDisk)
{
	printf("eta : %10le, %10le\n", gasDisk->eta.x, gasDisk->eta.y);
	printf("rho : %10le, %10le\n", gasDisk->rho.x, gasDisk->rho.y);
	printf("sch : %10le, %10le\n", gasDisk->sch.x, gasDisk->sch.y);
	printf("tau : %10le, %10le\n", gasDisk->tau.x, gasDisk->tau.y);
	printf("mfp : %10le, %10le\n", gasDisk->mfp.x, gasDisk->mfp.y);
	printf("temp: %10le, %10le\n", gasDisk->temp.x, gasDisk->temp.y);

	printf("gas_decrease: %d\n", gasDisk->gas_decrease);
	printf("          t0: %10le\n", gasDisk->t0);
	printf("          t1: %10le\n", gasDisk->t1);
	printf("   e_folding_time: %10le\n", gasDisk->e_folding_time);

	printf("c_vth  : %10le\n", gasDisk->c_vth);
	printf("alpha  : %10le\n", gasDisk->alpha);
	printf("m_star : %10le\n", gasDisk->m_star);
	printf("mean_molecular_weight : %10le\n", gasDisk->mean_molecular_weight);
	printf("   particle_diameter : %10le\n", gasDisk->particle_diameter);

	printf("\n");
	printf("reduction_factor(0)   : %10le\n", reduction_factor(gasDisk, 0));
	printf("reduction_factor(100) : %10le\n", reduction_factor(gasDisk, 100));
	printf("reduction_factor(150) : %10le\n", reduction_factor(gasDisk, 150));
	printf("reduction_factor(200) : %10le\n", reduction_factor(gasDisk, 200));
	printf("reduction_factor(250) : %10le\n", reduction_factor(gasDisk, 250));

}

static __host__ __device__ 
var_t	calculate_gamma_stokes(var_t cd, var_t density, var_t radius)
{
	if (density == 0.0 || radius == 0.0) {
		return 0.0;
	}
	else {
		return (3.0/8.0)*cd/(density*radius);
	}
}

static __host__ __device__ 
var_t	calculate_gamma_epstein(var_t density, var_t radius)
{
	if (density == 0.0 || radius == 0.0) {
		return 0.0;
	}
	else {
		return 1.0/(density*radius);
	}
}

void	populate_ppd(pp_disk *ppd)
{
	vec_t *coor = (vec_t*)ppd->h_y[0].data();
	vec_t *velo = (vec_t*)ppd->h_y[1].data();
	pp_disk::param_t* param = (pp_disk::param_t*)ppd->h_p.data();

	var_t	cd = 0.0;

	int i = 0;
	param[i].id = i;
	param[i].mass = 1.0;
	param[i].radius = 0.0046491301477493566;
	param[i].density = 2375725.6065676026;
	param[i].gamma_stokes = calculate_gamma_stokes(0.0, param[i].density, param[i].radius); // 0
	param[i].gamma_epstein = calculate_gamma_epstein(param[i].density, param[i].radius); //	9.0538233626552165e-005

	param[i].migType = MIGRATION_TYPE_NO;
	param[i].migStopAt = 0.0;;

	coor[i].x = coor[i].y = coor[i].z = 0.0;
	velo[i].x = velo[i].y = velo[i].z = 0.0;

	i = 1;
	param[i].id = i;
	param[i].mass = 2.0109496206846280e-026;
	param[i].radius = 1.4181811719423849e-011;
	param[i].density = 1683129.1259964250;
	param[i].gamma_stokes = calculate_gamma_stokes(1.0, param[i].density, param[i].radius); // 15710.21458376437
	param[i].gamma_epstein = calculate_gamma_epstein(param[i].density, param[i].radius); // 41893.905556705002

	param[i].migType = MIGRATION_TYPE_NO;
	param[i].migStopAt = 0.0;;

	coor[i].x = 1.0;
	coor[i].y = coor[i].z = 0.0;
	velo[i].x = 0.0;
	velo[i].y = 0.01720209895; // Ezt az értéket pontositani
	velo[i].z = 0.0;
}

cudaError_t  unit_test_pp_disk()
{
	cudaError_t cudaStatus = cudaSuccess;

	bool	succeeded = true;
	char	func_name[256];
	char	err_msg[1024];

	// 
	{
		bool	failed = false;
		strcpy(func_name, "unit_test_cpy_gas_disc_to_dev");

		number_of_bodies* nBodies = new number_of_bodies(1, 0, 0, 0, 0, 1, 0);
		gas_disk *gasDisk = new gas_disk();

		pp_disk *ppd = new pp_disk(nBodies, gasDisk, 0.0);
		populate_ppd(ppd);
		ppd->copy_to_device();

		vec_t *coor = (vec_t*)ppd->d_y[0].data().get();
		vec_t *velo = (vec_t*)ppd->d_y[1].data().get();
		pp_disk::param_t* param = (pp_disk::param_t*)ppd->d_p.data().get();

		int ndim = sizeof(vec_t) / sizeof(var_t);
		//! Holds the derivatives for the differential equations
		d_var_t	d_f;
		d_f.resize(ndim * nBodies->n_gas_drag());
		vec_t *aGD = (vec_t*)d_f.data().get();

		// Calculate accelerations originated from gas drag
		cudaStatus = ppd->call_calculate_drag_accel_kernel(0, param, coor, velo, aGD);
		// print out the result:
		// thrust::copy(d_f.begin(), d_f.end(), std::ostream_iterator<int>(std::cout, "\n"));

		/* expected result:
		 * [0]	-0.0
		 * [1]	-3.9620870296390791e-008
		 * [2]	-0.0
		 */


	}

	return cudaStatus;
}


cudaError_t unit_test_cpy_gas_disc_to_dev()
{
	cudaError_t cudaStatus = cudaSuccess;

	bool	succeeded = true;
	char	func_name[256];
	char	err_msg[1024];

	{
		bool	failed = false;
		strcpy(func_name, "unit_test_cpy_gas_disc_to_dev");

		gas_disk*	gasDisk;
		gas_disk*	d_gasDisk;
		//gasDisk = new gas_disk();

		// Modify default values:
		var2_t eta = { 2.0e-3, 1.0/2.0 };
		var2_t rho = { 1.4e-9 * Constants::GramPerCm3ToSolarPerAu3, -11.0/4.0 };
		var2_t sch = { 2.0e-2, 5.0/4.0 };
		var2_t tau = { 2.0/3.0, 2.0    };
		gasDisk = new gas_disk(rho, sch, eta, tau, GAS_DENSITY_DECREASE_EXPONENTIAL, 100.0, 200.0, 50.0, 1.0);

		cout << "gasDisk: " << endl;
		cout << *gasDisk;

		cudaStatus = HANDLE_ERROR(cudaMalloc((void**)&d_gasDisk, sizeof(gas_disk)));
		if (cudaStatus != cudaSuccess) {
			sprintf(err_msg, "\t%30s() function failed at line %d.", func_name, __LINE__);
			cerr << err_msg << endl;
			failed = true;
		}

		cudaStatus = HANDLE_ERROR(cudaMemcpy(d_gasDisk, gasDisk, sizeof(gas_disk), cudaMemcpyHostToDevice ));
		if (cudaStatus != cudaSuccess) {
			sprintf(err_msg, "\t%30s() function failed at line %d.", func_name, __LINE__);
			cerr << err_msg << endl;
			failed = true;
		}

		print_gas_disc<<<1,1>>>(d_gasDisk);
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaStatus != cudaSuccess) {
			sprintf(err_msg, "\t%30s() function failed at line %d.", func_name, __LINE__);
			cerr << err_msg << endl;
			failed = true;
		}

		cudaFree(d_gasDisk);
		delete gasDisk;
	}

	return cudaStatus;
}

// a = a + b
__global__
void add_two_vector(int_t n, var_t *a, const var_t *b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (n > tid) {
		a[tid] += b[tid];
	}
}

cudaError_t unit_test_transform_plus()
{
	cudaError_t cudaStatus = cudaSuccess;

	bool	succeeded = true;
	char	func_name[256];
	char	err_msg[1024];

	{
		bool	failed = false;
		strcpy(func_name, "add_two_vector");

		h_var_t h_acce;
		h_var_t	h_acceGasDrag;
		h_acce.resize(10 * 4);
		h_acceGasDrag.resize(3 * 4);

		for (int i = 0; i < 10*4; i++ ) {
			h_acce[i] = 0.0;
		}

		for (int i = 0; i < 3*4; i++ ) {
			h_acceGasDrag[i] = 1.0;
		}

		d_var_t acce = h_acce;
		d_var_t	acceGasDrag = h_acceGasDrag;

		int_t n = acceGasDrag.size();
		// 1 star + 1 gp + 3 rp
		int offset = 5 * 4;

		add_two_vector<<<1, n>>>(n, (var_t*)(acce.data().get() + offset), (var_t*)acceGasDrag.data().get());
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaStatus != cudaSuccess) {
			sprintf(err_msg, "\t%30s() function failed at line %d.", func_name, __LINE__);
			cerr << err_msg << endl;
			failed = true;
		}

		h_acce = acce;
		for (int i = 0; i < 10; i++ ) {
			int idx = 4*i;
			printf("h_acce[%d] = %10lf, %10lf, %10lf, %10lf\n", idx, h_acce[idx], h_acce[idx+1], h_acce[idx+2], h_acce[idx+3]);
		}

	}

	return cudaStatus;
}

int main(int argc, const char** argv)
{
	cudaError_t cudaStatus = cudaSuccess;
	int		result = 0;
	char	func_name[256];
	char	err_msg[1024];

	{
		strcpy(func_name, "unit_test_cpy_gas_disc_to_dev");

		cudaStatus = unit_test_cpy_gas_disc_to_dev();
		if (cudaSuccess == cudaStatus) {
			sprintf(err_msg, "The unit test(s) of the %s() function passed.", func_name);
			cout << endl << err_msg << endl;
		}
		else {
			sprintf(err_msg, "The unit test(s) of the %s() function failed.", func_name);
			cout << endl << err_msg << endl;
		}
	}

	{
		strcpy(func_name, "unit_test_transform_plus");

		cudaStatus = unit_test_transform_plus();
		if (cudaSuccess == cudaStatus) {
			sprintf(err_msg, "The unit test(s) of the %s() function passed.", func_name);
			cout << endl << err_msg << endl;
		}
		else {
			sprintf(err_msg, "The unit test(s) of the %s() function failed.", func_name);
			cout << endl << err_msg << endl;
		}
	}
	
	{
		strcpy(func_name, "unit_test_pp_disk");

		cudaStatus = unit_test_pp_disk();
		if (cudaSuccess == cudaStatus) {
			sprintf(err_msg, "The unit test(s) of the %s() function passed.", func_name);
			cout << endl << err_msg << endl;
		}
		else {
			sprintf(err_msg, "The unit test(s) of the %s() function failed.", func_name);
			cout << endl << err_msg << endl;
		}
	}

	return result;
}
