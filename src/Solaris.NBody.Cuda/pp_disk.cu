// includes system 
#include <iomanip>
#include <iostream>
#include <fstream>

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/generate.h"
#include "thrust/copy.h"
#include "thrust/transform.h"

// includes project
#include "config.h"
#include "constants.h"
#include "gas_disk.h"
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "pp_disk.h"

using namespace std;

#define THREADS_PER_BLOCK	256

#ifdef STOP_WATCH

void pp_disk::clear_elasped()
{
	for (int i = 0; i < PP_DISK_KERNEL_N; i++)
	{
		elapsed[i] = 0.0;
	}
}

string pp_disk::kernel_name[PP_DISK_KERNEL_N] = {
	"PP_DISK_KERNEL_THRUST_COPY_FROM_DEVICE_TO_DEVICE",
	"PP_DISK_KERNEL_THRUST_COPY_TO_DEVICE",
	"PP_DISK_KERNEL_THRUST_COPY_TO_HOST",
	"PP_DISK_KERNEL_ADD_TWO_VECTOR",
	"PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_TRIAL",
	"PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL",
	"PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_SELF_INTERACTING",
	"PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_SELF_INTERACTING",
	"PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_INTERACTING",
	"PP_DISK_KERNEL_CALCULATE_DRAG_ACCEL",
	"PP_DISK_KERNEL_CALCULATE_MIGRATEI_ACCEL",
	"PP_DISK_KERNEL_CALCULATE_ORBELEM"
};

#endif

static cudaError_t HandleError(cudaError_t cudaStatus, const char *file, int line)
{
    if (cudaSuccess != cudaStatus) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( cudaStatus ), file, line );
        return cudaStatus;
    }
	return cudaStatus;
}
#define HANDLE_ERROR(cudaStatus) (HandleError(cudaStatus, __FILE__, __LINE__))

static 
var_t pdf_const(var_t x)
{
	return 1;
}

// Draw a number from a given distribution
static 
var_t generate_random(var_t xmin, var_t xmax, var_t p(var_t))
{
	var_t x;
	var_t y;

	do
	{
		x = xmin + (var_t)rand() / RAND_MAX * (xmax - xmin);
		y = (var_t)rand() / RAND_MAX;
	}
	while (y > p(x));

	return x;
}

static __host__ __device__
void shift_into_range(var_t lower, var_t upper, var_t* value)
{
    var_t range = upper - lower;
    while (upper <= *value) {
        *value -= range;
    }
    while (lower > *value) {
        *value += range;
    }
}

static __host__ __device__
vec_t	vector_subtract(const vec_t* a, const vec_t* b)
{
	vec_t result;

	result.x = a->x - b->x;
    result.y = a->y - b->y;
    result.z = a->z - b->z;

	return result;
}

static __host__ __device__
vec_t	cross_product(const vec_t* v, const vec_t* u)
{
	vec_t result;

	result.x = v->y*u->z - v->z*u->y;
    result.y = v->z*u->x - v->x*u->z;
    result.z = v->x*u->y - v->y*u->x;

	return result;
}

static __host__ __device__
var_t	dot_product(const vec_t* v, const vec_t* u)
{
	return v->x * u->x + v->y * u->y + v->z * u->z;
}

static __host__ __device__
var_t	norm2(const vec_t* v)
{
	return SQR(v->x) + SQR(v->y) + SQR(v->z);
}

static __host__ __device__
var_t	norm(const vec_t* v)
{
	return sqrt(norm2(v));
}

#define FOUR_PI_OVER_THREE	4.1887902047863909846168578443727
static __host__ __device__
var_t calculate_radius(var_t m, var_t density)
{
	return pow(1.0/FOUR_PI_OVER_THREE * m/density ,1.0/3.0);
}

static __host__ __device__
var_t calculate_density(var_t m, var_t R)
{
	return m / (FOUR_PI_OVER_THREE * CUBE(R));
}

static __host__ __device__
var_t caclulate_mass(var_t R, var_t density)
{
	return FOUR_PI_OVER_THREE * CUBE(R) * density;
}
#undef FOUR_PI_OVER_THREE

static __host__ __device__
vec_t	circular_velocity(var_t mu, const vec_t* rVec)
{
	vec_t result = {0.0, 0.0, 0.0, 0.0};

	var_t r		= sqrt(SQR(rVec->x) + SQR(rVec->y));
	var_t vc	= sqrt(mu/r);

	var_t p;
	if (rVec->x == 0.0 && rVec->y == 0.0) {
		return result;
	}
	else if (rVec->y == 0.0) {
		result.y = rVec->x > 0.0 ? vc : -vc;
	}
	else if (rVec->x == 0.0) {
		result.x = rVec->y > 0.0 ? -vc : vc;
	}
	else if (rVec->x >= rVec->y) {
		p = rVec->y / rVec->x;
		result.y = rVec->x >= 0 ? vc/sqrt(1.0 + SQR(p)) : -vc/sqrt(1.0 + SQR(p));
		result.x = -result.y*p;
	}
	else {
		p = rVec->x / rVec->y;
		result.x = rVec->y >= 0 ? -vc/sqrt(1.0 + SQR(p)) : vc/sqrt(1.0 + SQR(p));
		result.y = -result.x*p;
	}

	return result;
}

static __host__ __device__
vec_t	gas_velocity(var2_t eta, var_t mu, const vec_t* rVec)
{
	vec_t result = circular_velocity(mu, rVec);
	var_t r		= sqrt(SQR(rVec->x) + SQR(rVec->y));

	var_t v		 = sqrt(1.0 - 2.0*eta.x * pow(r, eta.y));
	result.x	*= v;
	result.y	*= v;
	
	return result;
}

// TODO: implemet INNER_EDGE to get it from the input
#define INNER_EDGE 0.1 // AU
static __host__ __device__
var_t	gas_density_at(const gas_disk* gasDisk, const vec_t* rVec)
{
	var_t result = 0.0;

	var_t r		= sqrt(SQR(rVec->x) + SQR(rVec->y));
	var_t h		= gasDisk->sch.x * pow(r, gasDisk->sch.y);
	var_t arg	= SQR(rVec->z/h);
	if (INNER_EDGE < r) {
		result	= gasDisk->rho.x * pow(r, gasDisk->rho.y) * exp(-arg);
	}
	else {
		var_t a	= gasDisk->rho.x * pow(INNER_EDGE, gasDisk->rho.y - 4.0);
		result	= a * SQR(SQR(r)) * exp(-arg);
	}

	return result;
}
#undef INNER_EDGE

static __host__ __device__
var_t	calculate_kinetic_energy(const vec_t* vVec)
{
	return 0.5 * norm2(vVec);
}

static __host__ __device__
var_t	calculate_potential_energy(var_t mu, const vec_t* rVec)
{
	return -mu / norm(rVec);
}

static __host__ __device__
var_t	calculate_energy(var_t mu, const vec_t* rVec, const vec_t* vVec)
{
	return calculate_kinetic_energy(vVec) + calculate_potential_energy(mu, rVec);
}

static __host__ __device__
int_t	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E)
{
	if (ecc == 0.0 || mean == 0.0 || mean == PI) {
        *E = mean;
		return 0;
    }
    *E = mean + ecc * (sin(mean)) / (1.0 - sin(mean + ecc) + sin(mean));
    var_t E1 = 0.0;
    var_t error;
    int_t step = 0;
    do {
        E1 = *E - (*E - ecc * sin(*E) - mean) / (1.0 - ecc * cos(*E));
        error = fabs(E1 - *E);
        *E = E1;
    } while (error > eps && step++ <= 15);
	if (step > 15 ) {
		return 1;
	}

	return 0;
}

static __host__ __device__
int_t	calculate_phase(var_t mu, const pp_disk::orbelem_t* oe, vec_t* rVec, vec_t* vVec)
{
    var_t ecc = oe->ecc;
	var_t E = 0.0;
	if (kepler_equation_solver(ecc, oe->mean, 1.0e-14, &E) == 1) {
		return 1;
	}
    var_t v = 2.0 * atan(sqrt((1.0 + ecc) / (1.0 - ecc)) * tan(E / 2.0));

    var_t p = oe->sma * (1.0 - SQR(ecc));
    var_t r = p / (1.0 + ecc * cos(v));
    var_t kszi = r * cos(v);
    var_t eta = r * sin(v);
    var_t vKszi = -sqrt(mu / p) * sin(v);
    var_t vEta = sqrt(mu / p) * (ecc + cos(v));

    var_t cw = cos(oe->peri);
    var_t sw = sin(oe->peri);
    var_t cO = cos(oe->node);
    var_t sO = sin(oe->node);
    var_t ci = cos(oe->inc);
    var_t si = sin(oe->inc);

    vec_t P;
	P.x = cw * cO - sw * sO * ci;
	P.y = cw * sO + sw * cO * ci;
	P.z = sw * si;
    vec_t Q;
	Q.x = -sw * cO - cw * sO * ci;
	Q.y = -sw * sO + cw * cO * ci;
	Q.z = cw * si;

	rVec->x = kszi * P.x + eta * Q.x;
	rVec->y = kszi * P.y + eta * Q.y;
	rVec->z = kszi * P.z + eta * Q.z;

	vVec->x = vKszi * P.x + vEta * Q.x;
	vVec->y = vKszi * P.y + vEta * Q.y;
	vVec->z = vKszi * P.z + vEta * Q.z;

	return 0;
}

#define	sq3	1.0e-14
static __host__ __device__
int_t	calculate_sma_ecc(var_t mu, const vec_t* rVec, const vec_t* vVec, var_t* sma, var_t* ecc)
{
	// Calculate energy, h
    var_t h = calculate_energy(mu, rVec, vVec);
    if (h >= 0.0) {
        return 1;
    }

	// Calculate semi-major axis, a
    *sma = -mu / (2.0 * h);

    vec_t cVec = cross_product(rVec, vVec);
	cVec.w = norm2(&cVec);		// cVec.w = c2

	// Calculate eccentricity, e
    var_t e2 = 1.0 + 2.0 * h * cVec.w / SQR(mu);
	*ecc = fabs(e2) < sq3 ? 0.0 : sqrt(e2); 

    return 0;
}
#undef	sq3

#define	sq2 1.0e-14
#define	sq3	1.0e-14
static __host__ __device__
int_t	calculate_orbelem(var_t mu, const vec_t* rVec, const vec_t* vVec, pp_disk::orbelem_t* oe)
{
	// Calculate energy, h
    var_t h = calculate_energy(mu, rVec, vVec);
    if (h >= 0.0) {
        return 1;
    }

	var_t r = norm(rVec);
	var_t v = norm(vVec);

	vec_t cVec	= cross_product(rVec, vVec);
	vec_t vxc	= cross_product(vVec, &cVec);
	vec_t lVec;
	lVec.x		= -mu/r * rVec->x + vxc.x;
	lVec.y		= -mu/r * rVec->y + vxc.y;
	lVec.z		= -mu/r * rVec->z + vxc.z;
	lVec.w		= norm(&lVec);

	cVec.w = norm2(&cVec);		// cVec.w = c2
    
    // Calculate eccentricity, e
	var_t ecc = 1.0 + 2.0 * h * cVec.w / SQR(mu);
	ecc = abs(ecc) < sq3 ? 0.0 : sqrt(ecc); 

	// Calculate semi-major axis, a
    var_t sma = -mu / (2.0 * h);

    // Calculate inclination, incl
	cVec.w = sqrt(cVec.w);		// cVec.w = c
    var_t cosi = cVec.z / cVec.w;
    var_t sini = sqrt(SQR(cVec.x) + SQR(cVec.y)) / cVec.w;
    var_t incl = acos(cosi);
    if (incl < sq2) {
        incl = 0.0;
    }
    
    // Calculate longitude of node, O
    var_t node = 0.0;
    if (incl != 0.0) {
		var_t tmpx = -cVec.y / (cVec.w * sini);
        var_t tmpy =  cVec.x / (cVec.w * sini);
		node = atan2(tmpy, tmpx);
		shift_into_range(0.0, 2.0*PI, &node);
    }
    
    // Calculate argument of pericenter, w
    var_t E		= 0.0;
    var_t peri	= 0.0;
    if (ecc != 0.0) {
		var_t tmpx = ( lVec.x * cos(node) + lVec.y * sin(node)) / lVec.w;
        var_t tmpy = (-lVec.x * sin(node) + lVec.y * cos(node)) /(lVec.w * cosi);
        peri = atan2(tmpy, tmpx);
        shift_into_range(0.0, 2.0*PI, &peri);

        tmpx = 1.0 / ecc * (1.0 - r / sma);
		tmpy = dot_product(rVec, vVec) / (sqrt(mu * sma) * ecc);
        E = atan2(tmpy, tmpx);
        shift_into_range(0.0, 2.0*PI, &E);
    }
    else {
        peri = 0.0;
        E = atan2(rVec->y, rVec->x);
        shift_into_range(0.0, 2.0*PI, &E);
    }
    
    // Calculate mean anomaly, M
    var_t M = E - ecc * sin(E);
    shift_into_range(0.0, 2.0*PI, &M);

	oe->sma	= sma;
	oe->ecc	= ecc;
	oe->inc	= incl;
	oe->peri= peri;
	oe->node= node;
	oe->mean= M;

	return 0;
}
#undef	sq2
#undef	sq3

static __host__ __device__
var_t	orbital_period(var_t mu, var_t sma)
{
	return TWOPI * sqrt(CUBE(sma)/mu);
}

static __host__ __device__
var_t	orbital_frequency(var_t mu, var_t sma) 
{
	return 1.0 / orbital_period(mu, sma);
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

static __host__ __device__
var_t	reduction_factor(const gas_disk* gasDisk, ttt_t t)
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

#define SQRT_TWO_PI	2.50663
static __host__ __device__
var_t midplane_density(const gas_disk* gasDisk, var_t r)
{
	var_t a1 = gasDisk->rho.x * pow(r, gasDisk->rho.y);
	var_t a2 = gasDisk->sch.x * pow(r, gasDisk->sch.y);
	var_t a3 = a1 * a2 * SQRT_TWO_PI;

	return a3;
}
#undef SQRT_TWO_PI	

static __host__ __device__
var_t typeI_migration_time(const gas_disk* gasDisk, var_t C, var_t O, var_t ar, var_t er, var_t h)
{
	var_t result = 0.0;

	var_t Cm = 2.0/(2.7 + 1.1 * abs(gasDisk->rho.y))/O;
	var_t er1 = er / (1.3*h);
	var_t er2 = er / (1.1*h);
	var_t frac = (1.0 + FIFTH(er1)) / (1.0 - FORTH(er2));
	result = Cm * C * SQR(ar) * frac;

	return result;
}

#define Q	0.78
static __host__ __device__
var_t typeI_eccentricity_damping_time(var_t C, var_t O, var_t ar, var_t er, var_t h)
{
	var_t result = 0.0;

	var_t Ce = 0.1 / (Q*O);
	var_t frac = 1.0 + 0.25 * CUBE(er/h);
	result = Ce * C * FORTH(ar) * frac;

	return result;
}
#undef Q

static __host__ __device__
var_t  mean_thermal_speed_CMU(const gas_disk* gasDisk, var_t r)
{
	return gasDisk->c_vth * sqrt(gasDisk->temp.x * pow(r, gasDisk->temp.y));
}

// Calculate acceleration caused by particle j on particle i 
static __host__ __device__ 
vec_t calculate_grav_accel_pair(const vec_t ci, const vec_t cj, var_t mass, vec_t ai)
{
	vec_t dVec;
	
	dVec.x = cj.x - ci.x;
	dVec.y = cj.y - ci.y;
	dVec.z = cj.z - ci.z;

	dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
	var_t r = sqrt(dVec.w);								// = r

	dVec.w = mass / (r*dVec.w);

	ai.x += dVec.w * dVec.x;
	ai.y += dVec.w * dVec.y;
	ai.z += dVec.w * dVec.z;

	return ai;
}

/****************** KERNEL functions starts here ******************/

// a = a + b
static __global__
void add_two_vector_kernel(int_t n, var_t *a, const var_t *b)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (n > tid) {
		a[tid] += b[tid];
	}
}

//__constant__ int_t kernel_param[4];
//
//static __global__
//void	calculate_grav_accel_trial_kernel(/*interaction_bound iBound, */const pp_disk::param_t* params, const vec_t* coor, vec_t* acce)
//{
//	//const int bodyIdx = iBound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;
//	const int bodyIdx = kernel_param[0] + blockIdx.x * blockDim.x + threadIdx.x;
//
//	__shared__ var_t data[20];
//
//	//vec_t ai;
//	//if (bodyIdx < iBound.sink.y) {
//	//if (bodyIdx < kernel_param[1]) {
//		//vec_t dVec; = data[0..3]
//		//vec_t ci = coor[bodyIdx];
//		data[4] = coor[bodyIdx].x;
//		data[5] = coor[bodyIdx].y;
//		data[6] = coor[bodyIdx].z;
//		//data[7] = coor[bodyIdx].w;
//
//		//ai = acce[bodyIdx]; = data[8..11]
//
//		data[8] = data[9] =  data[10] =  data[11] = 0.0;
//		//for (int j = kernel_param[2]; j < kernel_param[3]; j++) 
//		{
//			//if (j == bodyIdx) {
//			//	continue;
//			//}
//			int j = 1;
//	
//			data[0] = coor[j].x - data[4];
//			data[1] = coor[j].y - data[5];
//			data[2] = coor[j].z - data[6];;
//
//			data[3] = SQR(data[0]) + SQR(data[1]) + SQR(data[2]);	// = r2
//			//var_t r = sqrt(data[3]);								// = r
//			var_t r = 1;
//
//			data[3] = params[j].mass / (r*data[3]);
//
//			data[8 ] += data[3] * data[0];
//			data[9 ] += data[3] * data[1];
//			data[10] += data[3] * data[2];
//		}
//	//}
//	acce[bodyIdx].x = data[8];
//	acce[bodyIdx].y = data[9];
//	acce[bodyIdx].z = data[10];
//}

static __global__
void	calculate_grav_accel_kernel(interaction_bound iBound, const pp_disk::param_t* params, const vec_t* coor, vec_t* acce)
{
	const int bodyIdx = iBound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (bodyIdx < iBound.sink.y) {
		vec_t ai;
		vec_t dVec;
		vec_t ci = coor[bodyIdx];
		ai.x = ai.y = ai.z = ai.w = 0.0;
		for (int j = iBound.source.x; j < iBound.source.y; j++) 
		{
			if (j == bodyIdx) {
				continue;
			}
	
			dVec.x = coor[j].x - ci.x;
			dVec.y = coor[j].y - ci.y;
			dVec.z = coor[j].z - ci.z;

			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
			var_t r = sqrt(dVec.w);								// = r

			dVec.w = params[j].mass / (r*dVec.w);

			ai.x += dVec.w * dVec.x;
			ai.y += dVec.w * dVec.y;
			ai.z += dVec.w * dVec.z;
		}
		acce[bodyIdx].x = K2 * ai.x;
		acce[bodyIdx].y = K2 * ai.y;
		acce[bodyIdx].z = K2 * ai.z;
	}
}

static __global__
void	calculate_grav_accel_for_massive_bodies_kernel(interaction_bound iBound, int nuw, const pp_disk::param_t* params, const vec_t* coor, vec_t* acce)
{
	const int sinkIdx	= iBound.sink.x + blockIdx.y;
	int sourceIdx		= iBound.source.x + threadIdx.x * nuw;
	// TODO: 32 helyett dinamikusan kiszamolni 2 megfelelő hatványát
	// Ez függeni fog a szálak számától, azaz a blockdim.x - től
	__shared__ vec_t partial_ai[32];

	partial_ai[threadIdx.x].x = 0.0;
	partial_ai[threadIdx.x].y = 0.0;
	partial_ai[threadIdx.x].z = 0.0;
	// TODO: fölösleges az if, hiszen pont annyi blokkom van ahány sink-em
	if (sinkIdx < iBound.sink.y)
	{
		vec_t ai = {0.0, 0.0, 0.0, 0.0};
		while (sourceIdx < iBound.source.y) {
			vec_t dVec;
			vec_t ci = coor[sinkIdx];
			for (int j = sourceIdx; j < sourceIdx + nuw && j < iBound.source.y; j++) 
			{
				if (j == sinkIdx) {
					continue;
				}
	
				dVec.x = coor[j].x - ci.x;
				dVec.y = coor[j].y - ci.y;
				dVec.z = coor[j].z - ci.z;

				dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
				var_t r = sqrt(dVec.w);								// = r

				dVec.w = params[j].mass / (r*dVec.w);

				ai.x += dVec.w * dVec.x;
				ai.y += dVec.w * dVec.y;
				ai.z += dVec.w * dVec.z;
			}
			sourceIdx += blockDim.x * nuw;
		}
		partial_ai[threadIdx.x] = ai;
		__syncthreads();

		int i = 32/2;
		int cacheIdx = threadIdx.x;
		while (i != 0)
		{
			if (cacheIdx < i)
			{
				partial_ai[cacheIdx].x += partial_ai[cacheIdx + i].x;
				partial_ai[cacheIdx].y += partial_ai[cacheIdx + i].y;
				partial_ai[cacheIdx].z += partial_ai[cacheIdx + i].z;
			}
			__syncthreads();
			i /= 2;
		}
		if (threadIdx.x == 0)
		{
			acce[sinkIdx].x = K2 * partial_ai[0].x;
			acce[sinkIdx].y = K2 * partial_ai[0].y;
			acce[sinkIdx].z = K2 * partial_ai[0].z;
		}
	}
}

static __global__
void calculate_drag_accel_kernel(interaction_bound iBound, var_t rFactor, const gas_disk* gasDisk, 
		const pp_disk::param_t* params, const vec_t* coor, const vec_t* velo, vec_t* acce)
{
	int tid		= blockIdx.x * blockDim.x + threadIdx.x;
	int	bodyIdx = iBound.sink.x + tid;

	if (bodyIdx < iBound.sink.y) {
		vec_t vGas	 = gas_velocity(gasDisk->eta, K2*params[0].mass, (vec_t*)&coor[bodyIdx]);
		var_t rhoGas = rFactor * gas_density_at(gasDisk, (vec_t*)&coor[bodyIdx]);
		var_t r = norm((vec_t*)&coor[bodyIdx]);

		vec_t u;
		u.x	= velo[bodyIdx].x - vGas.x;
		u.y	= velo[bodyIdx].y - vGas.y;
		u.z	= velo[bodyIdx].z - vGas.z;
		var_t C	= 0.0;

		var_t lambda = gasDisk->mfp.x * pow(r, gasDisk->mfp.y);
		// Epstein-regime:
		if (     params[bodyIdx].radius <= 0.1 * lambda)
		{
			var_t vth = mean_thermal_speed_CMU(gasDisk, r);
			C = params[bodyIdx].gamma_epstein * vth * rhoGas;
		}
		// Stokes-regime:
		else if (params[bodyIdx].radius >= 10.0 * lambda)
		{
			C = params[bodyIdx].gamma_stokes * norm(&u) * rhoGas;
		}
		// Transition-regime:
		else
		{

		}

		acce[tid].x = -C * u.x;
		acce[tid].y = -C * u.y;
		acce[tid].z = -C * u.z;
		acce[tid].w = 0.0;

		//printf("acce[tid].x: %10le\n", acce[tid].x);
		//printf("acce[tid].y: %10le\n", acce[tid].y);
		//printf("acce[tid].z: %10le\n", acce[tid].z);
	}
}

static __global__
void calculate_migrateI_accel_kernel(interaction_bound iBound, var_t rFactor, const gas_disk* gasDisk, 
		pp_disk::param_t* params, const vec_t* coor, const vec_t* velo, vec_t* acce)
{
	int tid		= blockIdx.x * blockDim.x + threadIdx.x;
	int	bodyIdx = iBound.sink.x + tid;

	var_t r = norm((vec_t*)&coor[bodyIdx]);
	if (params[bodyIdx].migStopAt > r) {
		acce[tid].x = acce[tid].y = acce[tid].z = acce[tid].w = 0.0;
		params[bodyIdx].migType = MIGRATION_TYPE_NO;
	}

	if (bodyIdx < iBound.sink.y && params[bodyIdx].migType == MIGRATION_TYPE_TYPE_I) {
		var_t a = 0.0, e = 0.0;
		var_t mu = K2*(params[0].mass + params[bodyIdx].mass);
		calculate_sma_ecc(mu, (vec_t*)(&coor[bodyIdx]), (vec_t*)(&velo[bodyIdx]), &a, &e);

		// Orbital frequency: (note, that this differs from the formula of Fogg & Nelson 2005)
		var_t O = orbital_frequency(mu, a); // K * sqrt((params[0]->mass + p->mass)/CUBE(a));
		var_t C = SQR(params[0].mass)/(params[bodyIdx].mass * SQR(a) * midplane_density(gasDisk, r));
		// Aspect ratio:
		var_t h = gasDisk->sch.x * pow(r, gasDisk->sch.y);
		var_t ar = h/r;
		var_t er = e*r;

		/*
		 *  When e > 1.1 h/r, inward migration halts as $t_{\rm m}$ becomes negative and only
		 *  resumes when eccentricity is damped to lower values. We note that under certain
		 *  circumstances, such as there being a surface density jump, or an optically thick disk,
		 *  or MHD turbulence, type I migration may be substantially modified or reversed
		 *  (Papaloizou & Nelson 2005; Paardekooper & Mellema 2006; Nelson 2005; Masset et al. 2006).
		 */
		var_t tm = 0.0;
		if (e < 1.1*ar) {
			tm = typeI_migration_time(gasDisk, C, O, ar, er, h);
			tm = 1.0/tm;
		}
		var_t te = typeI_eccentricity_damping_time(C, O, ar, er, h);
		var_t ti = te;
		var_t vr = dot_product((vec_t*)&coor[bodyIdx], (vec_t*)&velo[bodyIdx]);
		te = 2.0*vr/(r*r*te);
		ti = 2.0/ti;

		acce[tid].x = -rFactor*(tm * velo[bodyIdx].x + te * coor[bodyIdx].x);
		acce[tid].y = -rFactor*(tm * velo[bodyIdx].y + te * coor[bodyIdx].y);
		acce[tid].z = -rFactor*(tm * velo[bodyIdx].z + te * coor[bodyIdx].z + ti * velo[bodyIdx].z);
		acce[tid].w = 0.0;
	}
}

static __global__
void	calculate_orbelem_kernel(
		int_t total, int_t refBodyId, 
		const pp_disk::param_t *params, const vec_t *coor, const vec_t *velo, 
		pp_disk::orbelem_t *orbelem)
{
	int	bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (total > bodyIdx && refBodyId != bodyIdx) {
		var_t mu = K2 * (params[refBodyId].mass + params[bodyIdx].mass);
		vec_t rVec = vector_subtract((vec_t*)(&coor[bodyIdx]), (vec_t*)(&coor[refBodyId]));
		vec_t vVec = vector_subtract((vec_t*)(&velo[bodyIdx]), (vec_t*)(&velo[refBodyId]));

		calculate_orbelem(mu, &rVec, &vVec, (pp_disk::orbelem_t*)(&orbelem[bodyIdx]));
	}
}

/****************** KERNEL functions ends here ******************/




pp_disk::pp_disk(number_of_bodies *nBodies, gas_disk *gasDisk, ttt_t t0) :
	ode(2, t0),
	nBodies(nBodies),
	h_gasDisk(gasDisk),
	d_gasDisk(0),
	acceGasDrag(d_var_t()),
	acceMigrateI(d_var_t()),
	acceMigrateII(d_var_t())
{
	allocate_vectors();
#ifdef STOP_WATCH
	clear_elasped();
#endif
}

pp_disk::~pp_disk()
{
	delete nBodies;
	delete h_gasDisk;
	cudaFree(d_gasDisk);
}

void pp_disk::allocate_vectors()
{
	// Parameters
	int	npar = sizeof(param_t) / sizeof(var_t);
	h_p.resize(npar * nBodies->total);

	// Aliases to coordinates and velocities
	int ndim = sizeof(vec_t) / sizeof(var_t);
	h_y[0].resize(ndim * nBodies->total);
	h_y[1].resize(ndim * nBodies->total);

	if (0 != h_gasDisk) {
		acceGasDrag.resize(ndim * nBodies->n_gas_drag());
		// TODO:
		// ask Laci, how to find out if there was an error during these 2 cuda function calls
		cudaMalloc((void**)&d_gasDisk, sizeof(gas_disk));
		cudaMemcpy(d_gasDisk, h_gasDisk, sizeof(gas_disk), cudaMemcpyHostToDevice );

		if (0 < (nBodies->rocky_planet + nBodies->proto_planet)) {
			acceMigrateI.resize(ndim * (nBodies->rocky_planet + nBodies->proto_planet));
		}

		if (0 < nBodies->giant_planet) {
			acceMigrateII.resize(ndim * nBodies->giant_planet);
		}
	}
}

void pp_disk::calculate_grav_accel(interaction_bound iBound, const param_t* params, const vec_t* coor, vec_t* acce)
{
	for (int bodyIdx = iBound.sink.x; bodyIdx < iBound.sink.y; bodyIdx++) {
		vec_t ai = {0.0, 0.0, 0.0, 0.0};
		for (int j = iBound.source.x; j < iBound.source.y; j++) {
			if (j == bodyIdx) 
			{
				continue;
			}

			vec_t dVec;
			dVec.x = coor[j].x - coor[bodyIdx].x;
			dVec.y = coor[j].y - coor[bodyIdx].y;
			dVec.z = coor[j].z - coor[bodyIdx].z;

			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
			var_t r = sqrt(dVec.w);								// = r

			dVec.w = params[j].mass / (r*dVec.w);

			ai.x += dVec.w * dVec.x;
			ai.y += dVec.w * dVec.y;
			ai.z += dVec.w * dVec.z;

		}
		acce[bodyIdx].x = K2 * ai.x;
		acce[bodyIdx].y = K2 * ai.y;
		acce[bodyIdx].z = K2 * ai.z;
	}
}

cudaError_t pp_disk::call_calculate_grav_accel_kernel(const param_t *params, const vec_t *coor, vec_t *acce)
{
	cudaError_t cudaStatus = cudaSuccess;

	int		nBodyToCalculate;
	int		nThread;
	int		nBlock;
	
	nBodyToCalculate = nBodies->n_self_interacting();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_self_interacting();
		nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		grid.x		= nBlock;
		block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(iBound, params, coor, acce);
#if 0
		int	nuw		= 16;
		grid.x		= 1;
		grid.y		= nBodyToCalculate;
		grid.z		= 1;
		int nblock	= (iBound.source.y - iBound.source.x + nuw - 1)/nuw;
		block.x		= std::min(nblock, 32);
		block.y		= block.z = 1;

		calculate_grav_accel_for_massive_bodies_kernel<<<grid, block>>>(iBound, nuw, params, coor, acce);
#endif
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_SELF_INTERACTING] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_grav_accel_kernel failed", cudaStatus);
		}
	}

	nBodyToCalculate = nBodies->super_planetesimal + nBodies->planetesimal;
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_nonself_interacting();
		nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		grid.x		= nBlock;
		block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(iBound, params, coor, acce);
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_SELF_INTERACTING] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_grav_accel_kernel failed", cudaStatus);
		}
	}

	nBodyToCalculate = nBodies->test_particle;
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_non_interacting();
		nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		grid.x		= nBlock;
		block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(iBound, params, coor, acce);
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_INTERACTING] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_grav_accel_kernel failed", cudaStatus);
		}
	}

	return cudaStatus;
}

cudaError_t pp_disk::call_calculate_drag_accel_kernel(ttt_t time, const param_t *params, 
	const vec_t *coor, const vec_t *velo, vec_t *acce)
{
	cudaError_t cudaStatus = cudaSuccess;

	var_t timeF = reduction_factor(h_gasDisk, time);

	int	nBodyToCalculate = nBodies->n_gas_drag();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_bodies_gasdrag();
		int		nThread = std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		int		nBlock = (nBodyToCalculate + nThread - 1)/nThread;
		dim3	grid(nBlock);
		dim3	block(nThread);

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_drag_accel_kernel<<<grid, block>>>(iBound, timeF, d_gasDisk, params, coor, velo, acce);
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_DRAG_ACCEL] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_grav_accel_kernel failed", cudaStatus);
		}
	}
	return cudaStatus;
}

cudaError_t pp_disk::call_calculate_migrateI_accel_kernel(ttt_t time, param_t* params, 
	const vec_t* coor, const vec_t* velo, vec_t* acce)
{
	cudaError_t cudaStatus = cudaSuccess;

	var_t timeF = reduction_factor(h_gasDisk, time);

	int	nBodyToCalculate = nBodies->n_migrate_typeI();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_bodies_migrate_typeI();
		int		nThread = std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		int		nBlock = (nBodyToCalculate + nThread - 1)/nThread;
		dim3	grid(nBlock);
		dim3	block(nThread);

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_migrateI_accel_kernel<<<grid, block>>>(iBound, timeF, d_gasDisk, params, coor, velo, acce);
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_MIGRATEI_ACCEL] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_migrateI_accel_kernel failed", cudaStatus);
		}
	}
	return cudaStatus;
}

void pp_disk::calculate_dy(int i, int r, ttt_t t, const d_var_t& p, const std::vector<d_var_t>& y, d_var_t& dy)
{
	cudaError_t cudaStatus = cudaSuccess;

	switch (i)
	{
	case 0:
		// Copy velocities from previous step
#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		thrust::copy(y[1].begin(), y[1].end(), dy.begin());
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_THRUST_COPY_FROM_DEVICE_TO_DEVICE] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("thrust::copy kernel failed", cudaStatus);
		}
		break;
	case 1:
		// Make some shortcuts / aliases
		param_t* params = (param_t*)p.data().get();
		vec_t* coor		= (vec_t*)y[0].data().get();
		vec_t* velo		= (vec_t*)y[1].data().get();
		vec_t* acce		= (vec_t*)dy.data().get();
		// Calculate accelerations originated from the gravitational force
		call_calculate_grav_accel_kernel(params, coor, acce);

		if (0 != h_gasDisk && 0 < nBodies->n_gas_drag()) {
			vec_t *aGD = (vec_t*)acceGasDrag.data().get();
			if (0 == r) {
				// Calculate accelerations originated from gas drag
				call_calculate_drag_accel_kernel(t, params, coor, velo, aGD);
				cudaStatus = HANDLE_ERROR(cudaGetLastError());
				if (cudaSuccess != cudaStatus) {
					throw nbody_exception("call_calculate_drag_accel_kernel failed", cudaStatus);
				}
			}
			// Add acceGasDrag to dy
			int_t	offset = 4 * nBodies->n_self_interacting();
			var_t*	aSum = (var_t*)acce + offset;

			int	nData	= 4 * nBodies->n_gas_drag();
			int	nThread	= std::min(THREADS_PER_BLOCK, nData);
			int	nBlock	= (nData + nThread - 1)/nThread;
			dim3 grid(nBlock);
			dim3 block(nThread);

#ifdef STOP_WATCH
			s_watch.cuda_start();
#endif
			add_two_vector_kernel<<<grid, block>>>(nData, aSum, (var_t*)aGD);
#ifdef STOP_WATCH
			s_watch.cuda_stop();
			elapsed[PP_DISK_KERNEL_ADD_TWO_VECTOR] = s_watch.get_cuda_ellapsed_time();
#endif
			cudaStatus = HANDLE_ERROR(cudaGetLastError());
			if (cudaStatus != cudaSuccess) {
				throw nbody_exception("add_two_vector_kernel failed", cudaStatus);
			}
		}
		break;
	}
}

pp_disk::h_orbelem_t pp_disk::calculate_orbelem(int_t refBodyId)
{
	static const int noe = sizeof(orbelem_t)/sizeof(var_t);

	cudaError_t cudaStatus = cudaSuccess;

	// There are noe orbital elements
	d_orbelem.resize(noe * nBodies->total);

	// Calculate orbital elements of the bodies
	int		nThread = std::min(THREADS_PER_BLOCK, nBodies->total);
	int		nBlock = (nBodies->total + nThread - 1)/nThread;
	dim3	grid(nBlock);
	dim3	block(nThread);

	param_t	*params = (param_t*)d_p.data().get();
	vec_t	*coor = (vec_t*)d_y[0].data().get();
	vec_t	*velo = (vec_t*)d_y[1].data().get();

#ifdef STOP_WATCH
	s_watch.cuda_start();
#endif
	calculate_orbelem_kernel<<<grid, block>>>(nBodies->total, refBodyId, params, coor, velo, d_orbelem.data().get());
#ifdef STOP_WATCH
	s_watch.cuda_stop();
	elapsed[PP_DISK_KERNEL_CALCULATE_ORBELEM] = s_watch.get_cuda_ellapsed_time();
#endif
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("calculate_orbelem_kernel failed", cudaStatus);
	}
	// Download the result from the device
	h_orbelem = d_orbelem;

	return h_orbelem;
}

var_t	pp_disk::get_total_mass()
{
	var_t totalMass = 0.0;

	param_t* param = (param_t*)h_p.data();
	for (int j = 0; j < nBodies->n_massive(); j++ ) {
		totalMass += param[j].mass;
	}

	return totalMass;
}

void	pp_disk::compute_bc(var_t M0, vec_t* R0, vec_t* V0)
{
	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();
	param_t* param = (param_t*)h_p.data();

	for (int j = 0; j < nBodies->n_massive(); j++ ) {
		R0->x += param[j].mass * coor[j].x;
		R0->y += param[j].mass * coor[j].y;
		R0->z += param[j].mass * coor[j].z;

		V0->x += param[j].mass * velo[j].x;
		V0->y += param[j].mass * velo[j].y;
		V0->z += param[j].mass * velo[j].z;
	}
	R0->x /= M0;	R0->y /= M0;	R0->z /= M0;
	V0->x /= M0;	V0->y /= M0;	V0->z /= M0;
}

void pp_disk::transform_to_bc()
{
	cout << "Transform to barycentric system started";

	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();

	var_t totalMass = get_total_mass();

	// Position and velocity of the system's barycenter
	vec_t R0 = {0.0, 0.0, 0.0, 0.0};
	vec_t V0 = {0.0, 0.0, 0.0, 0.0};

	compute_bc(totalMass, &R0, &V0);

	// Transform the bodies coordinates and velocities
	for (int j = 0; j < nBodies->n_total(); j++ ) {
		coor[j].x -= R0.x;		coor[j].y -= R0.y;		coor[j].z -= R0.z;
		velo[j].x -= V0.x;		velo[j].y -= V0.y;		velo[j].z -= V0.z;
	}

	cout << " ... finished" << endl;
}

void pp_disk::load(string filename, int n)
{
	cout << "Loading position started";

	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();
	param_t* param = (param_t*)h_p.data();

	ifstream input(filename.c_str());

	if (input) {
		int_t	migType = 0;
		var_t	cd = 0.0;
		ttt_t	time = 0.0;
        		
		for (int i = 0; i < n; i++) { 
			input >> param[i].id;
			input >> time;

			input >> param[i].mass;
			input >> param[i].radius;
			input >> param[i].density;
			input >> cd;
			param[i].gamma_stokes = calculate_gamma_stokes(cd, param[i].density, param[i].radius);
			param[i].gamma_epstein = calculate_gamma_epstein(param[i].density, param[i].radius);
			input >> migType;
			param[i].migType = static_cast<migration_type_t>(migType);
			input >> param[i].migStopAt;

			input >> coor[i].x;
			input >> coor[i].y;
			input >> coor[i].z;

			input >> velo[i].x;
			input >> velo[i].y;
			input >> velo[i].z;
        }
        input.close();
	}
	else {
		throw nbody_exception("Cannot open file.");
	}

	cout << " ... finished" << endl;
}

void pp_disk::generate_rand(var2_t disk)
{
	int_t bodyId = 0;

	param_t* params = (param_t*)h_p.data();
	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();

	vec_t rVec = {0.0, 0.0, 0.0, 0.0};
	vec_t vVec = {0.0, 0.0, 0.0, 0.0};
	var_t cd;
	// Output central mass
	for (int i = 0; i < nBodies->star; i++, bodyId++)
	{
		params[i].id = bodyId;
		params[i].mass = 1.0;
		params[i].radius = Constants::SolarRadiusToAu;
		params[i].density = calculate_density(params[i].mass, params[i].radius);
		cd = 0.0;
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_NO;
		params[i].migStopAt = 0.0;

		coor[i] = rVec;
		velo[i] = vVec;
	}

	srand ((unsigned int)time(0));
	pp_disk::orbelem_t oe;
	// Output giant planets
	for (int i = 0; i < nBodies->giant_planet; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.1, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].mass = generate_random(0.1, 10.0, pdf_const) * Constants::JupiterToSolar;
		params[i].density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		params[i].radius = calculate_radius(params[i].mass, params[i].density);
		cd = 0.0;
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_TYPE_II;
		params[i].migStopAt = 1.0;

		var_t mu = K2*(params[0].mass + params[i].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	// Output rocky planets
	for (int i = 0; i < nBodies->rocky_planet; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.1, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].mass = generate_random(0.1, 10.0, pdf_const) * Constants::EarthToSolar;
		params[i].density = generate_random(3.0, 5.5, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		params[i].radius = calculate_radius(params[i].mass, params[i].density);
		cd = 0.0;
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_TYPE_I;
		params[i].migStopAt = 0.4;

		var_t mu = K2*(params[0].mass + params[i].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	// Output proto planets
	for (int i = 0; i < nBodies->proto_planet; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.1, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].mass = generate_random(0.001, 0.1, pdf_const) * Constants::EarthToSolar;
		params[i].density = generate_random(1.5, 3.5, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		params[i].radius = calculate_radius(params[i].mass, params[i].density);
		cd = 0.0;
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_TYPE_I;
		params[i].migStopAt = 0.4;

		var_t mu = K2*(params[0].mass + params[i].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	// Output super-planetesimals
	for (int i = 0; i < nBodies->super_planetesimal; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.2, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].mass = generate_random(0.0001, 0.01, pdf_const) * Constants::EarthToSolar;
		params[i].density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		params[i].radius = generate_random(5.0, 15.0, pdf_const) * Constants::KilometerToAu;
		cd = generate_random(0.5, 4.0, pdf_const);
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_NO;
		params[i].migStopAt = 0.0;

		var_t mu = K2*(params[0].mass + params[i].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	// Output planetesimals
	for (int i = 0; i < nBodies->planetesimal; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.2, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		params[i].radius = generate_random(5.0, 15.0, pdf_const) * Constants::KilometerToAu;
		params[i].mass = caclulate_mass(params[i].radius, params[i].density);
		cd = generate_random(0.5, 4.0, pdf_const);
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_NO;
		params[i].migStopAt = 0.0;

		var_t mu = K2*(params[0].mass + params[i].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	// Output test particles
	for (int i = 0; i < nBodies->test_particle; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.2, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		params[i].id = bodyId;
		params[i].density = 0.0;
		params[i].radius = 0.0;
		params[i].mass = 0.0;
		cd = 0.0;
		params[i].gamma_stokes = calculate_gamma_stokes(cd, params[i].density, params[i].radius);
		params[i].gamma_epstein = calculate_gamma_epstein(params[i].density, params[i].radius);
		params[i].migType = MIGRATION_TYPE_NO;
		params[i].migStopAt = 0.0;

		var_t mu = K2*(params[0].mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			throw nbody_exception("Cannot calculate phase.");
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}
}

// Print body positions
int pp_disk::print_positions(ostream& sout)
{
	cout << "Printing position started";

	param_t* h_param = (param_t*)h_p.data();
	vec_t* h_coord = (vec_t*)h_y[0].data();
	vec_t* h_veloc = (vec_t*)h_y[1].data();
	
	for (int i = 0; i < nBodies->total; i ++)
	{
		sout << h_param[i].id << '\t';
		sout << t << '\t';
		sout << h_param[i].mass << '\t';
		sout << h_param[i].radius << '\t';
		sout << h_param[i].density << '\t';
		sout << h_coord[i].x << '\t';
		sout << h_coord[i].y << '\t';
		sout << h_coord[i].z << '\t';
		sout << h_veloc[i].x << '\t';
		sout << h_veloc[i].y << '\t';
		sout << h_veloc[i].z;

		sout << endl;
	}

	cout << " ... finished" << endl;

	return 0;
}

// Print body orbital elements
int pp_disk::print_orbelem(ostream& sout)
{
	cout << "Printing orbital elements started";

	param_t *h_param = (param_t*)h_p.data();
	orbelem_t *oe	 = (orbelem_t*)h_orbelem.data();
	
	for (int i = 0; i < nBodies->total; i ++)
	{
		sout << h_param[i].id << '\t';
		sout << t << '\t';
		sout << h_param[i].mass << '\t';
		sout << h_param[i].radius << '\t';
		sout << h_param[i].density << '\t';
		sout << oe[i].sma << '\t';
		sout << oe[i].ecc << '\t';
		sout << oe[i].inc << '\t';
		sout << oe[i].peri << '\t';
		sout << oe[i].node << '\t';
		sout << oe[i].mean;

		sout << endl;
	}

	cout << " ... finished" << endl;

	return 0;
}
