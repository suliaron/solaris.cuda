// includes, system 
#include <ctime>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32 || _WIN64
#include <Windows.h>
#else
#include <sys/time.h>
#endif

// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes Thrust
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

// includes project
#include "config.h"
#include "constants.h" 
#include "gas_disk.h"
#include "nbody.h"
#include "nbody_exception.h"
#include "number_of_bodies.h"
#include "ode.h"
#include "options.h"
#include "pp_disk.h"

#include "timer.h"


using namespace std;

typedef enum device_type
{
	CPU,
	GPU
} device_type_t;


static cudaError_t HandleError(cudaError_t cudaStatus, const char *file, int line)
{
    if (cudaSuccess != cudaStatus) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( cudaStatus ), file, line );
        return cudaStatus;
    }
	return cudaStatus;
}
#define HANDLE_ERROR(cudaStatus) (HandleError(cudaStatus, __FILE__, __LINE__))

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

string create_number_of_bodies_str(const number_of_bodies *nBodies)
{
	ostringstream converter;   // stream used for the conversion
	
	converter << nBodies->star << '_' 
		<< nBodies->giant_planet << '_' 
		<< nBodies->rocky_planet << '_' 
		<< nBodies->proto_planet << '_' 
		<< nBodies->super_planetesimal << '_' 
		<< nBodies->planetesimal << '_' 
		<< nBodies->test_particle;

	return converter.str();
}

string combine_path(string dir, string filename)
{
	if (dir.size() > 0) {
		if (*(dir.end() - 1) != '/' && *(dir.end() - 1) != '\\') {
			return dir + '/' + filename;
		}
		else {
			return dir + filename;
		}
	}
	else {
		return filename;
	}
}

string get_filename(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
	}

	return result;
}

string get_filename_without_ext(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(pos + 1);
		pos = result.find_last_of('.');
		result = result.substr(0, pos);
	}

	return result;
}

string get_directory(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of("/\\");
		result = path.substr(0, pos);
	}

	return result;
}

string get_extension(const string& path)
{
	string result;

	if (path.size() > 0)
	{
		size_t pos = path.find_last_of('.');
		result = path.substr(pos + 1);
	}

	return result;
}


/* these functions were copied from pp_disk.cu */
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
/* copy-paste ends here */

var_t pdf_const(var_t x)
{
	return 1;
}

// Draw a number from a given distribution
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

#define FOUR_PI_OVER_THREE	4.1887902047863909846168578443727
var_t calculate_radius(var_t m, var_t density)
{
	return pow(1.0/FOUR_PI_OVER_THREE * m/density ,1.0/3.0);
}

var_t calculate_density(var_t m, var_t R)
{
	return m / (FOUR_PI_OVER_THREE * CUBE(R));
}

var_t caclulate_mass(var_t R, var_t density)
{
	return FOUR_PI_OVER_THREE * CUBE(R) * density;
}
#undef FOUR_PI_OVER_THREE


int populate_pp_disk(var2_t disk, const number_of_bodies *nBodies, pp_disk *ppd)
{
	int_t bodyId = 0;

	pp_disk::param_t* params = (pp_disk::param_t*)ppd->h_p.data();
	vec_t* coor = (vec_t*)(ppd->h_y[0]).data();
	vec_t* velo = (vec_t*)(ppd->h_y[1]).data();

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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
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
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		coor[i] = rVec;
		velo[i] = vVec;
	}

	return 0;
}


ttt_t compute_gravity_acceleration(number_of_bodies *nBodies, int_t iterMax, device_type_t dev_t)
{
	timer tmr;
	var_t result = 0.0;

	pp_disk *ppd = new pp_disk(nBodies, 0, 0.0);
	var2_t disk = {5.0, 6.0};	// AU
	ppd->generate_rand(disk);

	if (dev_t == GPU) 
	{
		ppd->copy_to_device();

		for (int i = 0; i < iterMax; i++) {
			tmr.cuda_start();
			ppd->calculate_dy(1, 0, 0.0, ppd->d_p, ppd->d_y, ppd->d_yout[1]);
			tmr.cuda_stop();
			result += (var_t)tmr.cuda_ellapsed_time();
		}
	}
	else {

		std::vector<var_t> h_acce;
		h_acce.resize(ppd->h_y[0].size());
		pp_disk::param_t* params = (pp_disk::param_t*)ppd->h_p.data();
		vec_t* coor = (vec_t*)ppd->h_y[0].data();
		vec_t* velo = (vec_t*)ppd->h_y[1].data();
		vec_t* h_a  = (vec_t*)h_acce.data();

		tmr.start();
		for (int i = 0; i < iterMax; i++) {
			if (0 < nBodies->n_self_interacting()) {
				interaction_bound iBound = nBodies->get_self_interacting();
				ppd->calculate_grav_accel(iBound, params, coor, h_a);
			}
			if (0 < nBodies->super_planetesimal + nBodies->planetesimal) {
				interaction_bound iBound	= nBodies->get_nonself_interacting();
				ppd->calculate_grav_accel(iBound, params, coor, h_a);
			}
			if (0 < nBodies->test_particle) {
				interaction_bound iBound = nBodies->get_non_interacting();
				ppd->calculate_grav_accel(iBound, params, coor, h_a);
			}
		}
		tmr.stop();
		result = tmr.ellapsed_time() / 1000.0;
	}

	delete ppd;

	return result / iterMax;
}

void parse_options(int argc, const char** argv, number_of_bodies **nBodies, int *iterMax, device_type_t *dev_t)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

		// Number of bodies
		if (p == "-nBodies") {
			i++;
			int	star				= atoi(argv[i++]);
			int	giant_planet		= atoi(argv[i++]);
			int	rocky_planet		= atoi(argv[i++]);
			int	proto_planet		= atoi(argv[i++]);
			int	super_planetesimal	= atoi(argv[i++]);
			int	planetesimal		= atoi(argv[i++]);
			int	test_particle		= atoi(argv[i]);
			*nBodies = new number_of_bodies(star, giant_planet, rocky_planet, proto_planet, super_planetesimal, planetesimal, test_particle);
		}
		else if (p == "-iM" ||p == "--iterMax") {
			i++;
			*iterMax = atoi(argv[i]);
		}
		else if (p == "-d" || p == "--device") {
			i++;
			p = argv[i];
			if (p == "CPU" || p == "cpu") {
				*dev_t = CPU;
			}
			else if (p == "GPU" || p == "gpu") {
				*dev_t = GPU;
			}
			else {
				cerr << "Invalid device: " << p << endl;
				exit(0);
			}
		}
		else {
			cerr << "Invalid switch on command-line." << endl;
			exit(0);
		}
		i++;
	}
}

// -n0 10000 -n1 10000 -dn 10 -dev gpu
int main(int argc, const char** argv)
{
	device_type_t dev_t = CPU;
	number_of_bodies *nBodies = 0;
	int_t iterMax = 10;

	parse_options(argc, argv, &nBodies, &iterMax, &dev_t);
	string dev_str = (dev_t == CPU ? "CPU_" : "GPU_kernel3_");

	//{
	//	string outDir = "D:\\Work\\Projects\\solaris.cuda\\PerformanceTest\\Debug";
	//	string outDir = "D:\\Work\\Projects\\solaris.cuda\\PerformanceTest\\Release";

	//	ofstream data;
	//	int_t iterMax = 10;
	//	for (int n = n0; n <= n1; n += dn) {
	//		for (int i = 1; i <= iterMax; i++) {
	//			number_of_bodies *nBodies = new number_of_bodies(0, n, 0, 0, 0, 0, 0);
	//			string filename = "gravity_acceleration_on_" + dev_str;
	//			filename += "nBodies_" + create_number_of_bodies_str(nBodies) + ".txt";

	//			string path = combine_path(outDir, filename);
	//			data.open(path.c_str(), std::ofstream::app);
	//			if (!data.is_open())
	//			{
	//				cerr << "Unable to open file: " << path << "!\n";
	//				return 0;
	//			}
	//			if ( i == 1 ) {
	//				data << "col1: # of iteration col2: execution time [msec]" << endl;
	//			}
	//			var_t elapsedTime = compute_gravity_acceleration(nBodies, i, dev_t);
	//			cout << setw(10) << i << " " << setw(10) << elapsedTime << endl;
	//			data << setw(10) << i << " " << setw(10) << elapsedTime << endl;
	//			data.close();
	//		}
	//	}
	//}

	{
		//string outDir = "D:\\Work\\Projects\\solaris.cuda\\PerformanceTest\\Debug";
		string outDir = "D:\\Work\\Projects\\solaris.cuda\\PerformanceTest\\Release";

		ofstream data;
		for (int n = n0; n <= n1; n += dn) {
			for (int i = 1; i <= iterMax; i++) {
				number_of_bodies *nBodies = new number_of_bodies(0, n, 0, 0, 0, 0, 10000);
				string filename = "gravity_acceleration_on_" + dev_str;
				filename += "nBodies_" + create_number_of_bodies_str(nBodies) + ".txt";

				string path = combine_path(outDir, filename);
				data.open(path.c_str(), std::ofstream::app);
				if (!data.is_open())
				{
					cerr << "Unable to open file: " << path << "!\n";
					return 0;
				}
				if ( i == 1 ) {
					data << "col1: # of iteration col2: execution time [msec]" << endl;
				}
				var_t elapsedTime = compute_gravity_acceleration(nBodies, i, dev_t);
				cout << setw(10) << i << " " << setw(10) << elapsedTime << endl;
				data << setw(10) << i << " " << setw(10) << elapsedTime << endl;
				data.close();
			}
		}
	}

	return 0;
}
