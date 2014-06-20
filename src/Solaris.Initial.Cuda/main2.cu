// includes, system 
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

// includes, project
#include "constants.h"
#include "number_of_bodies.h"
#include "pp_disk.h"

#define MASS_STAR		1.0			// M_sol
#define MASS_JUPITER	1.0e-3		// M_sol
#define RAD_STAR		0.005		// AU

#define MASS_SUN		1.9891E+30	// kg
#define MASS_FACTOR		5e-7		// M_sol
#define MASS_MU			log(4.0)
#define MASS_S			0.3
#define MASS_MIN		1.0e-20		// M_sol
#define MASS_MAX		1.0e-19		// M_sol

#define DIST_MIN		4.5			// AU
#define DIST_MAX		15			// AU

#define DENSITY			3000.0		// kg m-3
#define AU				149.6e9		// m

// It must be enclosed in parentheses in order to give correct results in
// the case of a division i.e. 1/SQR(x) -> 1/((x)*(x))
#define	SQR(x)			((x)*(x))
#define	CUBE(x)			((x)*(x)*(x))

using namespace std;

typedef enum output_version
		{
			FIRST_VERSION,
			SECOND_VERSION
		} output_version_t;

typedef struct oe_range
		{
			var2_t	sma;
			var_t  (*sma_p)(var_t);
			var2_t	ecc;
			var_t  (*inc_p)(var_t);
			var2_t	inc;
			var_t  (*ecc_p)(var_t);
			var2_t	peri;
			var_t  (*peri_p)(var_t);
			var2_t	node;
			var_t  (*node_p)(var_t);
			var2_t	mean;
			var_t  (*mean_p)(var_t);
		} oe_range_t;

typedef struct phys_prop_range
		{
			var2_t	mass;
			var_t  (*mass_p)(var_t);
			var2_t	radius;
			var_t  (*radius_p)(var_t);
			var2_t	density;
			var_t  (*density_p)(var_t);
			var2_t	cd;
			var_t  (*cd_p)(var_t);
		} phys_prop_range_t;

typedef struct body_disk
		{
			number_of_bodies	nBodies;
			oe_range_t			star_disk;
			phys_prop_range_t	star_pp;
			oe_range_t			gp_disk;
			phys_prop_range_t	gp_pp;
			oe_range_t			rp_disk;
			phys_prop_range_t	rp_pp;
			oe_range_t			pp_disk;
			phys_prop_range_t	pp_pp;
			oe_range_t			spl_disk;
			phys_prop_range_t	spl_pp;
			oe_range_t			pl_disk;
			phys_prop_range_t	pl_pp;
			oe_range_t			tp_disk;
		} body_disk_t;


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

var_t pdf_mass_lognormal(var_t x)
{
	return 1.0 / sqrt(2 * PI) / MASS_S * exp(-pow(log(x) - MASS_MU, 2) / 2 / MASS_S / MASS_MU);
}

var_t pdf_distance_squared(var_t d)
{
	return d * d / DIST_MAX / DIST_MAX;
}

var_t pdf_distance_exp(var_t d)
{
	return exp(-d) * d * d;
}

var_t pdf_const(var_t x)
{
	return 1;
}

var_t calculate_radius(var_t m)
{
	var_t V = m * MASS_SUN / DENSITY;	// m3
	V /= AU * AU * AU;		// AU3
	
	return pow(3.0 / 4.0 / PI * V, 1.0 / 3.0);
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


void print_body_record(std::ofstream &output, int_t bodyId, string name, var_t epoch, param_t *param, vec_t *r, vec_t *v, output_version_t o_version)
{
	static char sep = ' ';

	switch (o_version)
	{
	case FIRST_VERSION:
		output << bodyId << sep << epoch << sep;
		output << param->mass << sep << param->radius << sep << param->density << sep << param->cd << sep;
		output << param->migType << sep << param->migStopAt << sep;
		output << r->x << sep << r->y << sep << r->z << sep;
		output << v->x << sep << v->y << sep << v->z << sep;
		output << endl;
		break;
	case SECOND_VERSION:
		output << bodyId << sep << name << sep;

		output << param->mass << sep << param->radius << sep << param->density << sep << param->cd << sep;
		output << param->migType << sep << param->migStopAt << sep;
		output << r->x << sep << r->y << sep << r->z << sep;
		output << v->x << sep << v->y << sep << v->z << sep;
		output << endl;
		break;
	default:
		cerr << "Invalid output version!" << endl;
		exit(1);
	}
}

int generate_Dvorak_disk(string path, var2_t disk, number_of_bodies *nBodies)
{
	var_t t = 0.0;
	int_t bodyId = 0;

	var_t mCeres		= 9.43e20;	// kg
	var_t mMoon			= 9.43e20;	// kg
	var_t rhoBasalt		= 2.7;		// g / cm^3
	
	param_t param0;
	param_t param;
	vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
	vec_t	vVec = {0.0, 0.0, 0.0, 0.0};

	std::ofstream	output;
	output.open(path, std::ios_base::app);

	// Output central mass
	for (int i = 0; i < nBodies->star; i++, bodyId++)
	{
		param0.id = bodyId;
		param0.mass = 1.0;
		param0.radius = Constants::SolarRadiusToAu;
		param0.density = calculate_density(param0.mass, param0.radius);
		param0.cd = 0.0;
		param0.migType = NO;
		param0.migStopAt = 0.0;
		print_body_record(output, bodyId, "", t, &param0, &rVec, &vVec, FIRST_VERSION);
	}

	srand ((unsigned int)time(0));
	orbelem oe;
	// Output proto planets
	for (int i = 0; i < nBodies->proto_planet; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.3, pdf_const);
		oe.inc = 0.0;
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = 0.0;
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		param.id = bodyId;
		param.mass = generate_random(mCeres, mMoon / 10.0, pdf_const) * Constants::KilogramToSolar;
		param.density = rhoBasalt * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = calculate_radius(param.mass, param.density);
		param.cd = 0.0;
		param.migType = NO;
		param.migStopAt = 0.0;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
	}

	output.flush();
	output.close();

	return 0;
}

int __generate_pp_disk(string path, body_disk_t& body_disk)
{
	var_t t = 0.0;
	int_t bodyId = 0;


	pp_disk::param_t param0;
	pp_disk::param_t param;
	vec_t	rVec = {0.0, 0.0, 0.0, 0.0};
	vec_t	vVec = {0.0, 0.0, 0.0, 0.0};

	ofstream	output;
	output.open(path, std::ios_base::app);

	// Output central mass
	for (int i = 0; i < body_disk.nBodies.star; i++, bodyId++)
	{
		param0.id = bodyId;
		param0.mass = generate_random(body_disk.star_pp.mass.x, body_disk.star_pp.mass.y, body_disk.star_pp.mass_p);
		param0.radius = Constants::SolarRadiusToAu;
		param0.density = calculate_density(param0.mass, param0.radius);
		param0.cd = 0.0;
		param0.migType = NO;
		param0.migStopAt = 0.0;
		print_body_record(output, bodyId, "", t, &param0, &rVec, &vVec, FIRST_VERSION);
	}

	srand ((unsigned int)time(0));
	orbelem oe;
	// Output giant planets
	for (int i = 0; i < nBodies->giant_planet; i++, bodyId++)
	{
		oe.sma = generate_random(disk.x, disk.y, pdf_const);
		oe.ecc = generate_random(0.0, 0.1, pdf_const);
		oe.inc = atan(0.05); // tan(i) = h/r = 5.0e-2
		oe.peri = generate_random(0.0, 2.0*PI, pdf_const);
		oe.node = generate_random(0.0, 2.0*PI, pdf_const);
		oe.mean = generate_random(0.0, 2.0*PI, pdf_const);

		param.id = bodyId;
		param.mass = generate_random(0.1, 10.0, pdf_const) * Constants::JupiterToSolar;
		param.density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = calculate_radius(param.mass, param.density);
		param.cd = 0.0;
		param.migType = TYPE_II;
		param.migStopAt = 1.0;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
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

		param.id = bodyId;
		param.mass = generate_random(0.1, 10.0, pdf_const) * Constants::EarthToSolar;
		param.density = generate_random(3.0, 5.5, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = calculate_radius(param.mass, param.density);
		param.cd = 0.0;
		param.migType = TYPE_I;
		param.migStopAt = 0.4;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
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

		param.id = bodyId;
		param.mass = generate_random(0.001, 0.1, pdf_const) * Constants::EarthToSolar;
		param.density = generate_random(1.5, 3.5, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = calculate_radius(param.mass, param.density);
		param.cd = 0.0;
		param.migType = TYPE_I;
		param.migStopAt = 0.4;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
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

		param.id = bodyId;
		param.mass = generate_random(0.0001, 0.01, pdf_const) * Constants::EarthToSolar;
		param.density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = generate_random(5.0, 15.0, pdf_const) * Constants::KilometerToAu;
		param.cd = generate_random(0.5, 4.0, pdf_const);
		param.migType = NO;
		param.migStopAt = 0.0;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
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

		param.id = bodyId;
		param.density = generate_random(1.0, 2.0, pdf_const) * Constants::GramPerCm3ToSolarPerAu3;
		param.radius = generate_random(5.0, 15.0, pdf_const) * Constants::KilometerToAu;
		param.mass = caclulate_mass(param.radius, param.density);
		param.cd = generate_random(0.5, 4.0, pdf_const);
		param.migType = NO;
		param.migStopAt = 0.0;

		var_t mu = K2*(param0.mass + param.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
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

		param.id = bodyId;
		param.density = 0.0;
		param.radius = 0.0;
		param.mass = 0.0;
		param.cd = 0.0;
		param.migType = NO;
		param.migStopAt = 0.0;

		var_t mu = K2*(param0.mass);
		int_t ret_code = calculate_phase(mu, &oe, &rVec, &vVec);
		if (ret_code == 1) {
			cerr << "Could not calculate the phase." << endl;
			return ret_code;
		}
		print_body_record(output, bodyId, "", t, &param, &rVec, &vVec, FIRST_VERSION);
	}
	output.flush();
	output.close();

	return 0;
}

int parse_options(int argc, const char **argv, number_of_bodies **nBodies, string &outDir)
{
	int i = 1;

	while (i < argc) {
		string p = argv[i];

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
		else if (p == "-o") {
			i++;
			outDir = argv[i];
		}
		else {
			cerr << "Invalid switch on command-line.";
			return 1;
		}
		i++;
	}

	return 0;
}

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

int main(int argc, const char **argv)
{
	number_of_bodies *nBodies = 0;

	string outDir;
	int retCode = parse_options(argc, argv, &nBodies, outDir);
	if (0 != retCode) {
		exit(retCode);
	}
	string nbstr = create_number_of_bodies_str(nBodies);

	srand((unsigned int)time(NULL));

	//var2_t disk = {5.0, 6.0};	// AU
	//retCode = generate_pp_disk(combine_path(outDir, ("nBodies_" + nbstr + ".txt")), disk, nBodies);

	//var2_t disk = {65, 270};
	//retCode = generate_Rezso_disk(combine_path(outDir, ("nBodies_" + nbstr + ".txt")), disk, nBodies);

	// Generate Dvorak disk
	var2_t disk = {0.9, 2.5};	// AU
	retCode = generate_Dvorak_disk(combine_path(outDir, ("DvorakDisk01_" + nbstr + ".txt")), disk, nBodies);

	return retCode;
}
