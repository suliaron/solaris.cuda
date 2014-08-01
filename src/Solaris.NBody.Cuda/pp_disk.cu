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
#include "tools.h"

#define THREADS_PER_BLOCK	256

using namespace std;

__constant__ var_t d_cst_common[THRESHOLD_N];

#ifdef STOP_WATCH

void pp_disk::clear_elapsed()
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

static __global__
void initialize_event_data_t_kernel(int_t n_total, pp_disk::event_data_t* events, var_t value)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n_total)
	{
		events[tid].event_name = EVENT_NAME_NONE;
		events[tid].d = value;
	}
}

static __global__
	void check_collision_kernel(int_t n_total, const pp_disk::param_t* params, const pp_disk::event_data_t* pot_event, 
								pp_disk::event_data_t *occured_event, unsigned int *event_indexer)
{
	int bodyIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (params[bodyIdx].active && bodyIdx < n_total)
	{
		if (pot_event[bodyIdx].event_name == EVENT_NAME_CLOSE_ENCOUNTER)
		{
			int2_t idx = pot_event[bodyIdx].idx;
			var_t threshold_d = d_cst_common[THRESHOLD_COLLISION_FACTOR] * (params[idx.x].radius + params[idx.y].radius);
			bool collision = pot_event[bodyIdx].d < threshold_d ? true : false;
			if (collision && params[idx.y].active)
			{
				unsigned int i = atomicAdd(event_indexer, 1);
				//printf("d/threshold_d = %10lf %5d COLLISION detected. bodyIdx_0: %5d bodyIdx_1: %5d\n", pot_event[bodyIdx].d/threshold_d, i + 1, bodyIdx, pot_event[bodyIdx].idx.y);

				occured_event[i] = pot_event[bodyIdx];
				occured_event[i].event_name = EVENT_NAME_COLLISION;
			}
		}
	}
}

static __global__
	void check_hit_centrum_ejection_kernel(int_t n_total, ttt_t t, const vec_t* coor, const vec_t* velo, 
										   pp_disk::param_t* params, pp_disk::event_data_t *occured_event, unsigned int *event_indexer)
{
	int bodyIdx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (bodyIdx > 0 && params[bodyIdx].active && bodyIdx < n_total)
	{
		vec_t dVec;

		dVec.x = coor[bodyIdx].x - coor[0].x;
		dVec.y = coor[bodyIdx].y - coor[0].y;
		dVec.z = coor[bodyIdx].z - coor[0].z;
		dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2

		if (dVec.w > SQR(d_cst_common[THRESHOLD_EJECTION_DISTANCE]))
		{
			unsigned int i = atomicAdd(event_indexer, 1);
			//printf("r2 = %10lf\t%d EJECTION detected. bodyIdx: %d\n", dVec.w, i, bodyIdx);

			params[bodyIdx].active = false;

			occured_event[i].event_name = EVENT_NAME_EJECTION;
			occured_event[i].d = sqrt(dVec.w);
			occured_event[i].t = t;
			occured_event[i].id.x = 0;
			occured_event[i].id.y = params[bodyIdx].id;
			occured_event[i].idx.x = 0;
			occured_event[i].idx.y = bodyIdx;
			occured_event[i].r1 = coor[0];
			occured_event[i].v1 = velo[0];
			occured_event[i].r2 = coor[bodyIdx];
			occured_event[i].v2 = velo[bodyIdx];
		}
		else if (dVec.w < SQR(d_cst_common[THRESHOLD_HIT_CENTRUM_DISTANCE]))
		{
			unsigned int i = atomicAdd(event_indexer, 1);
			//printf("r2 = %10lf\t%d HIT_CENTRUM detected. bodyIdx: %d\n", dVec.w, i, bodyIdx);

			params[bodyIdx].active = false;

			occured_event[i].event_name = EVENT_NAME_HIT_CENTRUM;
			occured_event[i].d = sqrt(dVec.w);
			occured_event[i].t = t;
			occured_event[i].id.x = 0;
			occured_event[i].id.y = params[bodyIdx].id;
			occured_event[i].idx.x = 0;
			occured_event[i].idx.y = bodyIdx;
			occured_event[i].r1 = coor[0];
			occured_event[i].v1 = velo[0];
			occured_event[i].r2 = coor[bodyIdx];
			occured_event[i].v2 = velo[bodyIdx];
		}
	}
}

static __global__
	void	calculate_grav_accel_kernel(ttt_t t, interaction_bound iBound, const pp_disk::param_t* params, const vec_t* coor, const vec_t* velo, vec_t* acce, pp_disk::event_data_t* events)
{
	const int bodyIdx = iBound.sink.x + blockIdx.x * blockDim.x + threadIdx.x;

	if (params[bodyIdx].active && bodyIdx < iBound.sink.y) {
		vec_t ai;
		vec_t dVec;
		vec_t ci = coor[bodyIdx];
		ai.x = ai.y = ai.z = ai.w = 0.0;
		for (int j = iBound.source.x; j < iBound.source.y; j++) 
		{
			if (j == bodyIdx || !params[j].active) {
				continue;
			}
			// 3 FLOP
			dVec.x = coor[j].x - ci.x;
			dVec.y = coor[j].y - ci.y;
			dVec.z = coor[j].z - ci.z;

			// 5 FLOP
			dVec.w = SQR(dVec.x) + SQR(dVec.y) + SQR(dVec.z);	// = r2
			// TODO: use rsqrt()
			// 20 FLOP
			var_t r = sqrt(dVec.w);								// = r

			if (bodyIdx > 0 && bodyIdx < j && r < events[bodyIdx].d)
			{
				events[bodyIdx].event_name = EVENT_NAME_CLOSE_ENCOUNTER;
				events[bodyIdx].d = r;
				events[bodyIdx].t = t;
				events[bodyIdx].id.x = params[bodyIdx].id;
				events[bodyIdx].id.y = params[j].id;
				events[bodyIdx].idx.x = bodyIdx;
				events[bodyIdx].idx.y = j;
				events[bodyIdx].r1 = ci;
				events[bodyIdx].v1 = velo[bodyIdx];
				events[bodyIdx].r2 = coor[j];
				events[bodyIdx].v2 = velo[j];
			}
			// 2 FLOP
			dVec.w = params[j].mass / (r*dVec.w);

			// 6 FLOP
			ai.x += dVec.w * dVec.x;
			ai.y += dVec.w * dVec.y;
			ai.z += dVec.w * dVec.z;
		}
		// 36 FLOP
		acce[bodyIdx].x = K2 * ai.x;
		acce[bodyIdx].y = K2 * ai.y;
		acce[bodyIdx].z = K2 * ai.z;
	}

	// Try this because the stepsize was very small: 1.0e-10
	if (!params[bodyIdx].active)
	{
		acce[bodyIdx].x = 0.0;
		acce[bodyIdx].y = 0.0;
		acce[bodyIdx].z = 0.0;
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
	if (params[bodyIdx].mig_stop_at > r) {
		acce[tid].x = acce[tid].y = acce[tid].z = acce[tid].w = 0.0;
		params[bodyIdx].mig_type = MIGRATION_TYPE_NO;
	}

	if (bodyIdx < iBound.sink.y && params[bodyIdx].mig_type == MIGRATION_TYPE_TYPE_I) {
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


__global__
	void print_constant_mem_kernel()
{
	printf("print_constant_mem kernel:\n");
	printf("d_cst_common[THRESHOLD_HIT_CENTRUM_DISTANCE] : %lf\n", d_cst_common[THRESHOLD_HIT_CENTRUM_DISTANCE]);
	printf("d_cst_common[THRESHOLD_EJECTION_DISTANCE] : %lf\n", d_cst_common[THRESHOLD_EJECTION_DISTANCE]);
	printf("d_cst_common[THRESHOLD_COLLISION_FACTOR] : %lf\n", d_cst_common[THRESHOLD_COLLISION_FACTOR]);
}

__global__
	void print_param_kernel(int n, const pp_disk::param_t *params)
{
	printf("params[%d].id           : %20d\n", n, params[n].id);
	printf("params[%d].active       : %20s\n", n, params[n].active ? "true" : "false");
	printf("params[%d].body_type    : %20d\n", n, params[n].body_type);
	printf("params[%d].mass         : %20.15lf\n", n, params[n].mass);
	printf("params[%d].density      : %20.15lf\n", n, params[n].density);
	printf("params[%d].radius       : %20.15lf\n", n, params[n].radius);
	printf("params[%d].cd           : %20.15lf\n", n, params[n].cd);
	printf("params[%d].epoch        : %20.15lf\n", n, params[n].epoch);
	printf("params[%d].gamma_epstein: %20.15lf\n", n, params[n].gamma_epstein);
	printf("params[%d].gamma_stokes : %20.15lf\n", n, params[n].gamma_stokes);
	printf("params[%d].mig_type     : %20d\n", n, params[n].mig_type);
	printf("params[%d].mig_stop_at  : %20.15lf\n", n, params[n].mig_stop_at);
}

__global__
	void print_coor_velo_kernel(int n, const vec_t* coor, const vec_t* velo)
{
	printf("coor[%d].x: %20.15lf\n", n, coor[n].x);
	printf("coor[%d].y: %20.15lf\n", n, coor[n].y);
	printf("coor[%d].z: %20.15lf\n", n, coor[n].z);
	printf("velo[%d].x: %20.15lf\n", n, velo[n].x);
	printf("velo[%d].y: %20.15lf\n", n, velo[n].y);
	printf("velo[%d].z: %20.15lf\n", n, velo[n].z);
}

/****************** KERNEL functions ends here ******************/




pp_disk::pp_disk(number_of_bodies *nBodies, bool has_gas, ttt_t t0) :
	ode(2, t0),
	nBodies(nBodies),
	d_gasDisk(0),
	n_par(12), /* How to make it atomatic?? */
	n_var(sizeof(vec_t) / sizeof(var_t)),
	h_event_indexer(0),
	d_event_indexer(0),
	n_event(0),
	acceGasDrag(d_var_t()),
	acceMigrateI(d_var_t()),
	acceMigrateII(d_var_t())
{
	allocate_vectors(has_gas);
	cudaMalloc((void **)&d_event_indexer, sizeof(int));
#ifdef STOP_WATCH
	clear_elapsed();
#endif
}

pp_disk::~pp_disk()
{
	delete nBodies;
	delete h_gasDisk;
	cudaFree(d_gasDisk);
}

void pp_disk::allocate_vectors(bool has_gas)
{
	// Parameters
	h_p.resize(n_par * nBodies->total);

	h_y[0].resize(n_var * nBodies->total);
	h_y[1].resize(n_var * nBodies->total);

	h_potential_event.resize(nBodies->total);
	d_potential_event.resize(nBodies->total);
	// Maximum nBodies / 5 events can be detected simultaneously
	int max_n_event = nBodies->total / 5;
	h_occured_event.resize(max_n_event);
	d_occured_event.resize(max_n_event);

	if (has_gas) {
		if (0 < nBodies->n_gas_drag()) {
			acceGasDrag.resize(n_var * nBodies->n_gas_drag());
		}
		
		if (0 < (nBodies->rocky_planet + nBodies->proto_planet)) {
			acceMigrateI.resize(n_var * (nBodies->rocky_planet + nBodies->proto_planet));
		}

		if (0 < nBodies->giant_planet) {
			acceMigrateII.resize(n_var * nBodies->giant_planet);
		}
	}
}

void pp_disk::clear_event_indexer()
{
	h_event_indexer = 0;
	cudaMemcpy(d_event_indexer, &h_event_indexer, sizeof(h_event_indexer), cudaMemcpyHostToDevice);
}

void pp_disk::clear_param(param_t* p)
{
	p->active = false;
	p->cd = 0.0;
	p->density = 0.0;
	p->epoch = 0.0;
	p->gamma_epstein = 0.0;
	p->gamma_stokes = 0.0;
	p->mass = 0.0;
	p->mig_stop_at = 0.0;
	p->radius = 0.0;
}

void pp_disk::cpy_threshold_values(const var_t* h_cst_common)
{
	cudaMemcpyToSymbol(d_cst_common, h_cst_common, THRESHOLD_N * sizeof(var_t));
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

void pp_disk::set_kernel_launch_param(int n_data)
{
	int		n_thread = std::min(THREADS_PER_BLOCK, n_data);
	int		n_block = (n_data + n_thread - 1)/n_thread;

	grid.x	= n_block;
	block.x = n_thread;
}

void pp_disk::call_initialize_event_data_t_kernel(var_t value)
{
	int		nBodyToCalculate = nBodies->n_massive();

	set_kernel_launch_param(nBodyToCalculate);
	event_data_t* events = (event_data_t*)d_potential_event.data().get();

	initialize_event_data_t_kernel<<<grid, block>>>(nBodyToCalculate, events, value);
	cudaError_t  cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("initialize_event_data_t_kernel failed", cudaStatus);
	}
}

void pp_disk::call_check_collision_kernel()
{
	int	nBodyToCalculate = nBodies->total;

	param_t* params = (param_t*)d_p.data().get();
	event_data_t* potential_event = (event_data_t*)d_potential_event.data().get();
	event_data_t* occured_event = (event_data_t*)d_occured_event.data().get();

	set_kernel_launch_param(nBodyToCalculate);

	clear_event_indexer();
	check_collision_kernel<<<grid, block>>>(nBodyToCalculate, params, potential_event, occured_event, d_event_indexer);

	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("check_collision_kernel failed", cudaStatus);
	}

	cudaMemcpy(&n_event, d_event_indexer, sizeof(n_event), cudaMemcpyDeviceToHost);
}

void pp_disk::call_check_hit_centrum_ejection_kernel()
{
	int	nBodyToCalculate = nBodies->total;

	param_t* params		 = (param_t*)d_p.data().get();
	vec_t* coor			 = (vec_t*)d_y[0].data().get();
	vec_t* velo			 = (vec_t*)d_y[1].data().get();
	event_data_t* occured_event = (event_data_t*)d_occured_event.data().get();

	set_kernel_launch_param(nBodyToCalculate);

	clear_event_indexer();
	check_hit_centrum_ejection_kernel<<<grid, block>>>(nBodyToCalculate, t, coor, velo, params, occured_event, d_event_indexer);

	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("check_hit_centrum_ejection_kernel failed", cudaStatus);
	}

	cudaMemcpy(&n_event, d_event_indexer, sizeof(n_event), cudaMemcpyDeviceToHost);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
}

void pp_disk::call_calculate_grav_accel_kernel(ttt_t currt, const param_t *params, const vec_t *coor, const vec_t *velo, vec_t *acce, event_data_t* events)
{
	cudaError_t cudaStatus = cudaSuccess;

	int		nBodyToCalculate;
	
	nBodyToCalculate = nBodies->n_self_interacting();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_self_interacting();
		set_kernel_launch_param(nBodyToCalculate);
		//nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		//nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		//grid.x		= nBlock;
		//block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(currt, iBound, params, coor, velo, acce, events);
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
		set_kernel_launch_param(nBodyToCalculate);
		//nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		//nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		//grid.x		= nBlock;
		//block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(currt, iBound, params, coor, velo, acce, events);
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
		set_kernel_launch_param(nBodyToCalculate);
		//nThread		= std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		//nBlock		= (nBodyToCalculate + nThread - 1)/nThread;
		//grid.x		= nBlock;
		//block.x		= nThread;

#ifdef STOP_WATCH
		s_watch.cuda_start();
#endif
		calculate_grav_accel_kernel<<<grid, block>>>(currt, iBound, params, coor, velo, acce, events);
#ifdef STOP_WATCH
		s_watch.cuda_stop();
		elapsed[PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_INTERACTING] = s_watch.get_cuda_ellapsed_time();
#endif
		cudaStatus = HANDLE_ERROR(cudaGetLastError());
		if (cudaSuccess != cudaStatus) {
			throw nbody_exception("calculate_grav_accel_kernel failed", cudaStatus);
		}
	}
}

void pp_disk::call_calculate_drag_accel_kernel(ttt_t time, const param_t *params, 
	const vec_t *coor, const vec_t *velo, vec_t *acce)
{
	cudaError_t cudaStatus = cudaSuccess;

	var_t timeF = reduction_factor(h_gasDisk, time);

	int	nBodyToCalculate = nBodies->n_gas_drag();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_bodies_gasdrag();
		set_kernel_launch_param(nBodyToCalculate);
		//int		nThread = std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		//int		nBlock = (nBodyToCalculate + nThread - 1)/nThread;
		//dim3	grid(nBlock);
		//dim3	block(nThread);

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
}

void pp_disk::call_calculate_migrateI_accel_kernel(ttt_t time, param_t* params, 
	const vec_t* coor, const vec_t* velo, vec_t* acce)
{
	cudaError_t cudaStatus = cudaSuccess;

	var_t timeF = reduction_factor(h_gasDisk, time);

	int	nBodyToCalculate = nBodies->n_migrate_typeI();
	if (0 < nBodyToCalculate) {
		interaction_bound iBound = nBodies->get_bodies_migrate_typeI();
		set_kernel_launch_param(nBodyToCalculate);
		//int		nThread = std::min(THREADS_PER_BLOCK, nBodyToCalculate);
		//int		nBlock = (nBodyToCalculate + nThread - 1)/nThread;
		//dim3	grid(nBlock);
		//dim3	block(nThread);

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
}

void pp_disk::handle_hit_centrum_ejection()
{
	event_data_t* occured_event = (event_data_t*)h_occured_event.data();

	param_t* params		 = (param_t*)h_p.data();
	for (int i = 0; i < n_event; i++)
	{
		params[occured_event[i].idx.y].active = false;
		if (occured_event[i].event_name == EVENT_NAME_HIT_CENTRUM)
		{
			handle_collision_pair(&occured_event[i]);
		}
	}
}

void pp_disk::get_survivor_merger_idx(int2_t id, int *survivIdx, int *mergerIdx)
{
	int i;
	int2_t idx;

	param_t* params = (param_t*)h_p.data();
	for (i = 0; i < nBodies->total; i++)
	{
		if (params[i].id == id.x)
		{
			idx.x = i;
			break;
		}
	}
	i++;
	for ( ; i < nBodies->total; i++)
	{
		if (params[i].id == id.y)
		{
			idx.y = i;
		}
	}

	*survivIdx = idx.x;
	*mergerIdx = idx.y;
	if (params[*mergerIdx].mass > params[*survivIdx].mass)
	{
		*survivIdx = idx.y;
		*mergerIdx = idx.x;
	}
}

void pp_disk::handle_collision_pair(event_data_t* event_data)
{
	int survivIdx = 0;
	int mergerIdx = 0;

	get_survivor_merger_idx(event_data->id, &survivIdx, &mergerIdx);

	// get device pointer aliases to coordinates and velocities
	vec_t* coor = (vec_t*)d_y[0].data().get();
	vec_t* velo = (vec_t*)d_y[1].data().get();

	// get host and device pointer aliases to parameters
	param_t* h_params = (param_t*)h_p.data();
	param_t* d_params = (param_t*)d_p.data().get();

	// Calculate position and velocitiy of the new object
	vec_t r0;
	vec_t v0;
	calculate_phase_after_collision(h_params[event_data->idx.x].mass, h_params[event_data->idx.y].mass, &(event_data->r1), 
		&(event_data->v1),  &(event_data->r2), &(event_data->v2), r0, v0);

	// Calculate mass, volume, radius and density of the new object
	var_t mass	 = h_params[survivIdx].mass + h_params[mergerIdx].mass;
	var_t volume = 4.188790204786391 * (CUBE(h_params[mergerIdx].radius) + CUBE(h_params[survivIdx].radius));
	var_t radius = pow(0.238732414637843 * volume, 1.0/3.0);
	var_t density= mass / volume;

	// Update mass, density and radius of survivor
	h_params[survivIdx].mass	= mass;
	h_params[survivIdx].density = density;
	h_params[survivIdx].radius  = radius;
	h_params[survivIdx].gamma_epstein = calculate_gamma_epstein(density, radius);
	h_params[survivIdx].gamma_stokes  = calculate_gamma_stokes(h_params[survivIdx].cd, density, radius);

	// Clear parameters of the merged object
	clear_param(&h_params[mergerIdx]);

	// Copy the new position, velocitiy and parameters up to the device

	//cout << "Position and velocity of survivor before merge:" << endl;
	//print_coor_velo_kernel<<<1, 1>>>(survivIdx, coor, velo);
	//cudaDeviceSynchronize();

	// Copy position
	cudaMemcpy(&coor[survivIdx], &r0, n_var * sizeof(var_t), cudaMemcpyHostToDevice);
	cudaError_t cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}
	// Copy velocity
	cudaMemcpy(&velo[survivIdx], &v0, n_var * sizeof(var_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}

	//cout << "Position and velocity of survivor after merge:" << endl;
	//print_coor_velo_kernel<<<1, 1>>>(survivIdx, coor, velo);
	//cudaDeviceSynchronize();

	//cout << "Paramaters of survivor before merge:" << endl;
	//print_param_kernel<<<1, 1>>>(survivIdx, d_params);
	//cudaDeviceSynchronize();

	// Copy parameters of the survivor
	cudaMemcpy(&d_params[survivIdx], &h_params[survivIdx], n_par * sizeof(var_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}

	//cout << "Paramaters of survivor after merge:" << endl;
	//print_param_kernel<<<1, 1>>>(survivIdx, d_params);
	//cudaDeviceSynchronize();

	//cout << "Paramaters of merger before merge:" << endl;
	//print_param_kernel<<<1, 1>>>(mergerIdx, d_params);
	//cudaDeviceSynchronize();

	// Copy parameters of the merger
	cudaMemcpy(&d_params[mergerIdx], &h_params[mergerIdx], n_par * sizeof(var_t), cudaMemcpyHostToDevice);
	cudaStatus = HANDLE_ERROR(cudaGetLastError());
	if (cudaSuccess != cudaStatus) {
		throw nbody_exception("cudaMemcpy failed", cudaStatus);
	}

	//cout << "Paramaters of merger after merge:" << endl;
	//print_param_kernel<<<1, 1>>>(mergerIdx, d_params);
	//cudaDeviceSynchronize();
}

void pp_disk::calculate_phase_after_collision(var_t m0, var_t m1, const vec_t* r1, const vec_t* v1, const vec_t* r2, const vec_t* v2, vec_t& r0, vec_t& v0)
{
	const var_t M = m0 + m1;

	r0.x = (m0 * r1->x + m1 * r2->x) / M;
	r0.y = (m0 * r1->y + m1 * r2->y) / M;
	r0.z = (m0 * r1->z + m1 * r2->z) / M;

	v0.x = (m0 * v1->x + m1 * v2->x) / M;
	v0.y = (m0 * v1->y + m1 * v2->y) / M;
	v0.z = (m0 * v1->z + m1 * v2->z) / M;
}

void pp_disk::cpy_data_to_device_after_collision()
{
}

void pp_disk::handle_collision()
{
	event_data_t* occured_event = (event_data_t*)h_occured_event.data();

	param_t* params		 = (param_t*)h_p.data();
	for (int i = 0; i < n_event; i++)
	{
		handle_collision_pair(&occured_event[i]);
	}
}

void pp_disk::calculate_dy(int i, int r, ttt_t currt, const d_var_t& p, const std::vector<d_var_t>& y, d_var_t& dy)
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
		param_t* params		 = (param_t*)p.data().get();
		param_t* h_params	 = (param_t*)h_p.data();
		event_data_t* potential_event = (event_data_t*)d_potential_event.data().get();
		vec_t* coor			 = (vec_t*)y[0].data().get();
		vec_t* velo			 = (vec_t*)y[1].data().get();
		vec_t* acce			 = (vec_t*)dy.data().get();

		if (r == 0)
		{
			// Set the d field of the event_data_t struct to the threshold distance when collision must be looked for
			// This is set to the radius of the star enhanced by 1 %.
			var_t threshold_distance_for_close_encounter = 1.01 * h_params[0].radius;
			call_initialize_event_data_t_kernel(threshold_distance_for_close_encounter);
		}

		// Calculate accelerations originated from the gravitational force
		call_calculate_grav_accel_kernel(currt, params, coor, velo, acce, potential_event);

		if (0 != h_gasDisk && 0 < nBodies->n_gas_drag()) {
			vec_t *aGD = (vec_t*)acceGasDrag.data().get();
			if (0 == r) {
				// Calculate accelerations originated from gas drag
				call_calculate_drag_accel_kernel(currt, params, coor, velo, aGD);
				cudaStatus = HANDLE_ERROR(cudaGetLastError());
				if (cudaSuccess != cudaStatus) {
					throw nbody_exception("call_calculate_drag_accel_kernel failed", cudaStatus);
				}
			}
			// Add acceGasDrag to dy
			int_t	offset = 4 * nBodies->n_self_interacting();
			var_t*	aSum = (var_t*)acce + offset;

			int	nData	= 4 * nBodies->n_gas_drag();
			set_kernel_launch_param(nData);
			//int	nThread	= std::min(THREADS_PER_BLOCK, nData);
			//int	nBlock	= (nData + nThread - 1)/nThread;
			//dim3 grid(nBlock);
			//dim3 block(nThread);

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

unsigned int pp_disk::get_n_event()
{
	return n_event;
}

var_t pp_disk::get_mass_of_star()
{
	param_t* param = (param_t*)h_p.data();
	for (int j = 0; j < nBodies->n_massive(); j++ ) {
		if (param[j].id == 0)
		{
			return param[j].mass;
		}
	}
	return 0.0;
}

var_t pp_disk::get_total_mass()
{
	var_t totalMass = 0.0;

	param_t* param = (param_t*)h_p.data();
	for (int j = 0; j < nBodies->n_massive(); j++ ) {
		totalMass += param[j].mass;
	}

	return totalMass;
}

void pp_disk::compute_bc(var_t M0, vec_t* R0, vec_t* V0)
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
	cout << "Transforming to barycentric system ... ";

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

	cout << "done" << endl;
}

void pp_disk::load(string path, int n)
{
	cout << "Loading " << path << " ... ";

	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();
	param_t* param = (param_t*)h_p.data();

	fstream input(path.c_str(), ios_base::in);

	if (input) {
		int_t	mig_type = 0;
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
			input >> mig_type;
			param[i].mig_type = static_cast<migration_type_t>(mig_type);
			input >> param[i].mig_stop_at;

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
		throw nbody_exception("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}

void pp_disk::load(string& path)
{
	cout << "Loading " << path << " ... ";

	ifstream input(path.c_str());
	if (input) 
	{
		int ns, ngp, nrp, npp, nspl, npl, ntp;
		ns = ngp = nrp = npp = nspl = npl = ntp = 0;
		input >> ns;
		input >> ngp;
		input >> nrp;
		input >> npp;
		input >> nspl;
		input >> npl;
		input >> ntp;
	}
	else 
	{
		throw nbody_exception("Cannot open " + path + ".");
	}

	vec_t* coor = (vec_t*)h_y[0].data();
	vec_t* velo = (vec_t*)h_y[1].data();
	param_t* param = (param_t*)h_p.data();

	if (input) {
		int_t	type = 0;
		var_t	cd = 0.0;
		string	dummy;
        		
		for (int i = 0; i < nBodies->total; i++) { 
			param[i].active = true;
			// id
			input >> param[i].id;
			// name
			input >> dummy;
			body_names.push_back(dummy);
			// body type
			input >> type;
			param[i].body_type = static_cast<body_type_t>(type);
			// epoch
			input >> param[i].epoch;

			// mass
			input >> param[i].mass;
			// radius
			input >> param[i].radius;
			// density
			input >> param[i].density;
			// stokes constant
			input >> cd;
			param[i].gamma_stokes = calculate_gamma_stokes(cd, param[i].density, param[i].radius);
			param[i].gamma_epstein = calculate_gamma_epstein(param[i].density, param[i].radius);

			// migration type
			input >> type;
			param[i].mig_type = static_cast<migration_type_t>(type);
			// migration stop at
			input >> param[i].mig_stop_at;

			// position
			input >> coor[i].x;
			input >> coor[i].y;
			input >> coor[i].z;
			// velocity
			input >> velo[i].x;
			input >> velo[i].y;
			input >> velo[i].z;
        }
        input.close();
	}
	else {
		throw nbody_exception("Cannot open " + path + ".");
	}

	cout << "done" << endl;
}

void pp_disk::print_event_data(ostream& sout, ostream& log_f)
{
	static char sep = ' ';
	static char *e_names[] = {"NONE", "HIT_CENTRUM", "EJECTION", "CLOSE_ENCOUNTER", "COLLISION"};

	thrust::copy(d_occured_event.begin(), d_occured_event.begin() + n_event, h_occured_event.begin());
	event_data_t* occured_event = (event_data_t*)h_occured_event.data();

	param_t* params	= (param_t*)h_p.data();
	for (int i = 0; i < n_event; i++)
	{
		sout << setw(20) << setprecision(10) << occured_event[i].t << sep
			 << setw(16) << e_names[occured_event[i].event_name] << sep
			 << setw( 8) << occured_event[i].id.x << sep
			 << setw( 8) << occured_event[i].id.y << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.x].mass << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.x].density << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.x].radius << sep
			 << setw(20) << setprecision(10) << occured_event[i].r1.x << sep
			 << setw(20) << setprecision(10) << occured_event[i].r1.y << sep
			 << setw(20) << setprecision(10) << occured_event[i].r1.z << sep
			 << setw(20) << setprecision(10) << occured_event[i].v1.x << sep
			 << setw(20) << setprecision(10) << occured_event[i].v1.y << sep
			 << setw(20) << setprecision(10) << occured_event[i].v1.z << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.y].mass << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.y].density << sep
			 << setw(20) << setprecision(10) << params[occured_event[i].idx.y].radius << sep
			 << setw(20) << setprecision(10) << occured_event[i].r2.x << sep
			 << setw(20) << setprecision(10) << occured_event[i].r2.y << sep
			 << setw(20) << setprecision(10) << occured_event[i].r2.z << sep
			 << setw(20) << setprecision(10) << occured_event[i].v2.x << sep
			 << setw(20) << setprecision(10) << occured_event[i].v2.y << sep
			 << setw(20) << setprecision(10) << occured_event[i].v2.z << sep << endl;

		char time_stamp[20];
		get_time_stamp(time_stamp);
		log_f << time_stamp << sep << e_names[occured_event[i].event_name] << endl;
	}
}

int pp_disk::print_positions(ostream& sout)
{
	cout << "Printing position ... ";

	param_t* h_param = (param_t*)h_p.data();
	vec_t* h_coord = (vec_t*)h_y[0].data();
	vec_t* h_veloc = (vec_t*)h_y[1].data();
	
	sout << t << '\t';
	for (int i = 0; i < nBodies->total; i ++)
	{
		sout << h_param[i].id << '\t';
		sout << h_param[i].mass << '\t';
		sout << h_param[i].radius << '\t';
		sout << h_param[i].density << '\t';
		sout << h_coord[i].x << '\t';
		sout << h_coord[i].y << '\t';
		sout << h_coord[i].z << '\t';
		sout << h_veloc[i].x << '\t';
		sout << h_veloc[i].y << '\t';
		sout << h_veloc[i].z << '\t';
	}
	sout << endl;

	cout << "done" << endl;

	return 0;
}

// Print body orbital elements
int pp_disk::print_orbelem(ostream& sout)
{
	cout << "Printing orbital elements ... ";

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

	cout << "done" << endl;

	return 0;
}
