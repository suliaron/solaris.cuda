#pragma once

#include <string>

#include "config.h"
#include "interaction_bound.h"
#include "ode.h"
#include "options.h"

#ifdef STOP_WATCH
#include "stop_watch.h"
#endif

class number_of_bodies;
class gas_disk;

using namespace std;

typedef enum migration_type
{
	MIGRATION_TYPE_NO,
	MIGRATION_TYPE_TYPE_I,
	MIGRATION_TYPE_TYPE_II
} migration_type_t;

typedef enum body_type
{
	BODY_TYPE_STAR,
	BODY_TYPE_GIANTPLANET,
	BODY_TYPE_ROCKYPLANET,
	BODY_TYPE_PROTOPLANET,
	BODY_TYPE_SUPERPLANETESIMAL,
	BODY_TYPE_PLANETESIMAL,
	BODY_TYPE_TESTPARTICLE,
	BODY_TYPE_N
} body_type_t;

typedef enum orbelem_name
		{
			SMA,
			ECC,
			INC,
			PERI,
			NODE,
			MEAN
		} orbelem_name_t;

typedef enum phys_prop_name
		{
			MASS,
			RADIUS,
			DENSITY,
			DRAG_COEFF
		} phys_prop_name_t;

typedef enum event_name
		{
			EVENT_NAME_NONE,
			EVENT_NAME_HIT_CENTRUM,
			EVENT_NAME_EJECTION,
			EVENT_NAME_CLOSE_ENCOUNTER,
			EVENT_NAME_COLLISION,
			EVENT_NAME_N
		} event_name_t;

#ifdef STOP_WATCH
typedef enum pp_disk_kernel
{
	PP_DISK_KERNEL_THRUST_COPY_FROM_DEVICE_TO_DEVICE,
	PP_DISK_KERNEL_THRUST_COPY_TO_DEVICE,
	PP_DISK_KERNEL_THRUST_COPY_TO_HOST,
	PP_DISK_KERNEL_ADD_TWO_VECTOR,
	PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_TRIAL,
	PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL,
	PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_SELF_INTERACTING,
	PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_SELF_INTERACTING,
	PP_DISK_KERNEL_CALCULATE_GRAV_ACCEL_NON_INTERACTING,
	PP_DISK_KERNEL_CALCULATE_DRAG_ACCEL,
	PP_DISK_KERNEL_CALCULATE_MIGRATEI_ACCEL,
	PP_DISK_KERNEL_CALCULATE_ORBELEM,
	PP_DISK_KERNEL_N
} pp_disk_kernel_t;
#endif

class pp_disk : public ode
{
public:

#ifdef STOP_WATCH
	stop_watch		s_watch;
	var_t			elapsed[PP_DISK_KERNEL_N];
	static string	kernel_name[PP_DISK_KERNEL_N];

	void			clear_elapsed();
#endif

	// Type for parameters
	typedef struct param
	{
		//! Unique number to identify the object
		int_t id;
		//! Type of the body
		body_type_t body_type;
		//! Indicates whether the body is participating in the simulation or not (i.e. escaped)
		bool active;
		//! The initial conditions are valid for this epoch
		var_t epoch;
		//! Mass of body in M_sol
		var_t mass;
		//! Radius of body in AU
		var_t radius;
		//! Density of body in M_sol AU-3
		var_t density;
		//! Drag coefficient for the Stokes-type drag
		var_t cd;
		//! Used for the drag force
		var_t gamma_stokes;
		//! Used for the drag force
		var_t gamma_epstein;
		//! Type of the migration
		migration_type_t mig_type;
		//! The migration stop at this distance measured from the star
		var_t mig_stop_at;
	} param_t;

	typedef struct event_data
	{
		event_name_t	event_name;	//!< Name of the event
		ttt_t	t;			//!< Time of the event
		int2_t	id;			//!< ids of the bodies
		int2_t	idx;		//!< indicies of the bodies
		var_t	d;			//!< distance of the bodies
		vec_t	r1;			//!< Position of body 1
		vec_t	v1;			//!< Velocity of body 1
		vec_t	r2;			//!< Position of body 2
		vec_t	v2;			//!< Velocity of body 2
	} event_data_t;

	typedef struct orbelem
	{
		//! Semimajor-axis of the body
		var_t sma;
		//! Eccentricity of the body
		var_t ecc;
		//! Inclination of the body
		var_t inc;
		//! Argument of the pericenter
		var_t peri;
		//! Longitude of the ascending node
		var_t node;
		//! Mean anomaly
		var_t mean;
	} orbelem_t;

	//typedef thrust::host_vector<param_t>		h_param_t;
	//typedef thrust::device_vector<param_t>		d_param_t;

	typedef thrust::host_vector<orbelem_t>		h_orbelem_t;
	typedef thrust::device_vector<orbelem_t>	d_orbelem_t;

	typedef thrust::host_vector<event_data_t>	h_event_data_t;
	typedef thrust::device_vector<event_data_t>	d_event_data_t;

	vector<string>		body_names;

	int					n_par;					//!< The number of parameters in the param_t type
	int					n_var;					//!< The number of variables in the vec_t type

	unsigned int		h_event_indexer;
	unsigned int		*d_event_indexer;
	unsigned int		n_event;

	d_orbelem_t			d_orbelem;
	h_orbelem_t			h_orbelem;

	h_event_data_t		h_potential_event;		//!< Vector on the host containing data for possible events
	d_event_data_t		d_potential_event;		//!< Vector on the device containing data for possible events
	h_event_data_t		h_occured_event;		//!< Vector on the host containing data for occured events
	d_event_data_t		d_occured_event;		//!< Vector on the device containing data for occured events

	gas_disk			*h_gasDisk;
	gas_disk			*d_gasDisk;
	
	pp_disk(number_of_bodies *nBodies, bool has_gas, ttt_t t0);
	~pp_disk();

	h_orbelem_t calculate_orbelem(int_t refBodyId);

	void calculate_dy(int i, int r, ttt_t currt, const d_var_t& p, const std::vector<d_var_t>& y, d_var_t& dy);

	//! Loads the initial position and velocity of the bodies (first input version).
	/*   
		\param path the full path of the data file
		\param n the total number of the bodies
	*/
	void load(string path, int n);
	//! Loads the initial position and velocity of the bodies (second input version).
	/*   
		\param path the full path of the data file
	*/
	void load(string& path);

	void generate_rand(var2_t disk);
	//! Print all bodies' id, mass, radius, density, position and velocity vector
	int print_positions(ostream& sout);
	int print_orbelem(ostream& sout);
	void cpy_threshold_values(const var_t *h_cst_common);

	void transform_to_bc();
	//! Return the number of the occured event
	unsigned int get_n_event();

	//! Computes the total mass of the system
	var_t get_total_mass();
	//! Returns the mass of the central star (the id of the star must be 0!)
	var_t get_mass_of_star();
	//! Compute the position and velocity of the system's barycenter
	/*  
		\param M0 will contains the total mass of the system
		\param R0 will contain the position of the barycenter
		\param V0 will contain the velocity of the barycenter
	*/
	void compute_bc(var_t M0, vec_t* R0, vec_t* V0);

	//! Calls the cpu function that calculates the accelerations from gravitational
	/*  interactions.
		\param iBound containes the staring and ending indicies of the sink and source bodies
		\param params Vector of parameters of the bodies
		\param coor Vector of coordinates of the bodies
		\param acce Will hold the accelerations for each body
	*/
	void calculate_grav_accel(interaction_bound iBound, const param_t* params, const vec_t* coor, vec_t* acce);

	void call_check_hit_centrum_ejection_kernel();
	//! Checks whether any collisions have occured
	void call_check_collision_kernel();
	//! Set the d field of the event_data_t to value
	/*
		\param value The the d field will be set to this value
	*/
	void call_initialize_event_data_t_kernel(var_t value);

	void print_event_data(ostream& sout, ostream& log);
	void handle_hit_centrum_ejection();
	void handle_collision();

	void get_survivor_merger_idx(int2_t id, int *survivIdx, int *mergerIdx);
	void handle_collision_pair(event_data_t* event_data);
	void cpy_data_to_device_after_collision();

private:
	dim3	grid;
	dim3	block;

	number_of_bodies	*nBodies;

	d_var_t				acceGasDrag;
	d_var_t				acceMigrateI;
	d_var_t				acceMigrateII;

	void allocate_vectors(bool has_gas);
	void clear_event_indexer();
	void clear_param(param_t* p);

	void calculate_phase_after_collision(var_t m0, var_t m1, const vec_t* r1, const vec_t* v1, const vec_t* r2, const vec_t* v2, vec_t& r0, vec_t& v0);

	//! Sets the grid and block for the kernel launch
	void set_kernel_launch_param(int n_data);



	//! Calls the kernel that calculates the accelerations from gravitational
	/*  interactions.
		\param currt Time of the evaluation of the gravitational force
		\param params Vector of parameters of the bodies
		\param coor Vector of coordinates of the bodies
		\param velo Vector of velocities of the bodies
		\param acce Will hold the accelerations for each body
		\param events Will hold the data for collision, ejection and hit-centrum check
	*/
	void call_calculate_grav_accel_kernel(ttt_t currt, const param_t *params, const vec_t *coor, const vec_t *velo, vec_t *acce, event_data_t* events);
	//! Calls the kernel that calculates the acceleration due to drag force.
	/*
		\param time The actual time of the simulation
		\param params Vector of parameters of the bodies
		\param coor Vector of coordinates of the bodies
		\param velo Vector of velocities of the bodies
		\param acce Will hold the accelerations for each body
	*/
	void call_calculate_drag_accel_kernel(ttt_t time, const param_t* params, const vec_t* coor, const vec_t* velo, vec_t* acce);

	//! Calls the kernel that calculates the acceleration due to type I migration.
	/*
		\param time The actual time of the simulation
		\param params Vector of parameters of the bodies
		\param coor Vector of coordinates of the bodies
		\param velo Vector of velocities of the bodies
		\param acce Will hold the accelerations for each body
	*/
	void call_calculate_migrateI_accel_kernel(ttt_t time, param_t* params, const vec_t* coor, const vec_t* velo, vec_t* acce);
};

static __host__ __device__ void		shift_into_range(var_t lower, var_t upper, var_t* value);
static __host__ __device__ vec_t	cross_product(const vec_t* v, const vec_t* u);
static __host__ __device__ var_t	dot_product(const vec_t* v, const vec_t* u);
static __host__ __device__ var_t	norm2(const vec_t* v);
static __host__ __device__ var_t	norm(const vec_t* v);
static __host__ __device__ vec_t	circular_velocity(var_t mu, const vec_t* rVec);
static __host__ __device__ vec_t	gas_velocity(var2_t eta, var_t mu, const vec_t* rVec);
static __host__ __device__ var_t	gas_density_at(const gas_disk* gasDisk, const vec_t* rVec);
static __host__ __device__ var_t	calculate_kinetic_energy(const vec_t* vVec);
static __host__ __device__ var_t	calculate_potential_energy(var_t mu, const vec_t* rVec);
static __host__ __device__ var_t	calculate_energy(var_t mu, const vec_t* rVec, const vec_t* vVec);
static __host__ __device__ int_t	kepler_equation_solver(var_t ecc, var_t mean, var_t eps, var_t* E);
static __host__ __device__ int_t	calculate_phase(var_t mu, const pp_disk::orbelem_t* oe, vec_t* rVec, vec_t* vVec);
static __host__ __device__ int_t	calculate_sma_ecc(var_t mu, const vec_t* coor, const vec_t* velo, var_t* sma, var_t* ecc);
static __host__ __device__ int_t	calculate_orbelem(var_t mu, const vec_t* coor, const vec_t* velo, pp_disk::orbelem_t* orbelem);
static __host__ __device__ var_t	orbital_period(var_t mu, var_t sma);
static __host__ __device__ var_t	orbital_frequency(var_t mu, var_t sma);
static __host__ __device__ var_t	calculate_gamma_stokes(var_t cd, var_t density, var_t radius);
static __host__ __device__ var_t	calculate_gamma_epstein(var_t density, var_t radius);
static __host__ __device__ var_t	reduction_factor(const gas_disk* gasDisk, ttt_t t);
static __host__ __device__ var_t	midplane_density(const gas_disk* gasDisk, var_t r);
static __host__ __device__ var_t	typeI_migration_time(const gas_disk* gasDisk, var_t C, var_t O, var_t ar, var_t er, var_t h);
static __host__ __device__ var_t	typeI_eccentricity_damping_time(var_t C, var_t O, var_t ar, var_t er, var_t h);
